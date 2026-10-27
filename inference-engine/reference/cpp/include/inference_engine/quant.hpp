#pragma once

/** @file
 *  @brief W8A8 quantization: int8 weights and activations, int32 sums.
 *
 * Weights get one symmetric int8 scale per output row, fixed when the model
 * is loaded. Activations get one symmetric int8 scale per input row,
 * computed at run time from that row's largest magnitude. A projection then
 * sums int8 products in int32 and rescales the sum back to Scalar. Integer
 * sums do not depend on their order, and every float step is a single
 * correctly rounded operation, so any implementation that performs the same
 * steps reproduces these results bit for bit.
 */

#include "inference_engine/tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace inference_engine {

/** Largest int8 magnitude used; -128 is left out so the range is symmetric.
 */
inline constexpr std::int32_t kInt8Limit = 127;

/** An [out_features, in_features] weight stored as int8 with row scales.
 *
 * Row r of the real weight is approximately values row r times scales[r].
 *
 * @tparam Scalar Floating-point type of the scales.
 */
template <typename Scalar> struct QuantizedWeight {
    /// int8 values shaped [out_features, in_features].
    Tensor<std::int8_t> values;
    /// One scale per output row.
    std::vector<Scalar> scales;
};

/** Scale that maps a largest magnitude onto kInt8Limit.
 *
 * @tparam Scalar Floating-point value type.
 * @param max_magnitude Largest absolute value being quantized.
 * @return max_magnitude / 127, or 1 when every value is zero.
 */
template <typename Scalar> Scalar int8_scale(Scalar max_magnitude) {
    static_assert(std::is_floating_point_v<Scalar>,
                  "int8_scale requires a floating-point scalar");
    return max_magnitude > Scalar{0}
               ? max_magnitude / static_cast<Scalar>(kInt8Limit)
               : Scalar{1};
}

/** Round value / scale to the nearest integer, ties to even, in [-127, 127].
 *
 * @tparam Scalar Floating-point value type.
 * @param value Value to quantize.
 * @param scale Positive scale from int8_scale().
 * @return The int8 code for @p value.
 * @throws std::invalid_argument If @p value or @p scale is infinite or NaN,
 * which has no int8 code.
 */
template <typename Scalar>
std::int8_t quantize_int8(Scalar value, Scalar scale) {
    static_assert(std::is_floating_point_v<Scalar>,
                  "quantize_int8 requires a floating-point scalar");
    if (!std::isfinite(value) || !std::isfinite(scale)) {
        throw std::invalid_argument("quantize_int8 requires finite values");
    }
    assert(scale > Scalar{0});
    constexpr auto kLimit = static_cast<Scalar>(kInt8Limit);
    // nearbyint rounds in the default mode, ties to even, as the target's
    // float-to-integer conversion does.
    const Scalar rounded = std::nearbyint(value / scale);
    return static_cast<std::int8_t>(std::clamp(rounded, -kLimit, kLimit));
}

/** Largest absolute value in one row of a matrix.
 *
 * @tparam Scalar Floating-point value type.
 * @param matrix Valid two-dimensional tensor.
 * @param row Row index.
 * @return The row's largest magnitude.
 */
template <typename Scalar>
Scalar row_max_magnitude(const Tensor<Scalar>& matrix, std::size_t row) {
    const std::size_t columns = matrix.shape[1];
    Scalar largest{0};
    for (std::size_t column = 0; column < columns; ++column) {
        largest = std::max(largest,
                           std::fabs(matrix.values[(row * columns) + column]));
    }
    return largest;
}

/** Quantize every row of a matrix with its own scale.
 *
 * Weights are quantized once with this; linear() applies the same rule to
 * its input rows on every call.
 *
 * @tparam Scalar Floating-point value type.
 * @param matrix Matrix shaped [rows, columns].
 * @return int8 values and one scale per row.
 * @throws std::invalid_argument If @p matrix is not a valid matrix or holds
 * an infinite or NaN value.
 */
template <typename Scalar>
QuantizedWeight<Scalar> quantize_rows(const Tensor<Scalar>& matrix) {
    if (!valid_matrix(matrix)) {
        throw std::invalid_argument("quantize_rows requires a valid matrix");
    }
    const std::size_t rows = matrix.shape[0];
    const std::size_t columns = matrix.shape[1];
    QuantizedWeight<Scalar> quantized{
        Tensor<std::int8_t>{matrix.shape,
                            std::vector<std::int8_t>(matrix.values.size())},
        std::vector<Scalar>(rows)};
    for (std::size_t row = 0; row < rows; ++row) {
        const Scalar scale = int8_scale(row_max_magnitude(matrix, row));
        quantized.scales[row] = scale;
        for (std::size_t column = 0; column < columns; ++column) {
            const std::size_t index = (row * columns) + column;
            quantized.values.values[index] =
                quantize_int8(matrix.values[index], scale);
        }
    }
    return quantized;
}

} // namespace inference_engine
