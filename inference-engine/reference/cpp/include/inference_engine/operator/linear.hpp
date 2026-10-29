#pragma once

/** @file
 *  @brief Dense affine projection shared by the attention and MLP blocks.
 */

#include "inference_engine/quant.hpp"
#include "inference_engine/tensor.hpp"
#include "inference_engine/util.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>

namespace inference_engine {

namespace detail {

/** Add one bias value per output feature to every row, in place.
 *
 * @tparam Scalar Tensor value type.
 * @param output Rows shaped [rows, out_features].
 * @param bias Optional reference to a bias shaped [out_features].
 * @throws std::invalid_argument If the bias is not one value per output
 * feature.
 */
template <typename Scalar>
void add_bias(
    Tensor<Scalar>& output,
    std::optional<std::reference_wrapper<const Tensor<Scalar>>> bias) {
    if (!bias.has_value()) {
        return;
    }
    const Tensor<Scalar>& values = bias->get();
    const std::size_t out_features = output.shape[1];
    if (values.shape.size() != 1 || values.shape[0] != out_features ||
        !valid_tensor(values)) {
        throw std::invalid_argument(
            "linear bias must hold one value per output feature");
    }
    for (std::size_t row = 0; row < output.shape[0]; ++row) {
        for (std::size_t column = 0; column < out_features; ++column) {
            Scalar& value = output.values[(row * out_features) + column];
            value = safe_add(value, values.values[column]);
        }
    }
}

} // namespace detail

/** Apply a weight matrix and an optional bias to input rows.
 *
 * Computes input @ weight.T + bias, matching torch.nn.Linear: @p weight
 * holds one row per output feature, so a [out_features, in_features] matrix
 * maps [rows, in_features] input to [rows, out_features] output. Passing a
 * weight in the opposite layout is rejected by matmul rather than silently
 * transposing the projection. Each output sums its products in order over
 * in_features, then adds the bias.
 *
 * @tparam Scalar Tensor value type.
 * @param input Input rows shaped [rows, in_features].
 * @param weight Output weights shaped [out_features, in_features].
 * @param bias Optional reference to a bias shaped [out_features]; no copy is
 *        made.
 * @return Projected rows shaped [rows, out_features].
 * @throws std::invalid_argument If the shapes or value counts are
 * incompatible, or the bias is not one value per output feature.
 * @throws std::overflow_error If a product or sum cannot be represented.
 */
template <typename Scalar>
Tensor<Scalar>
linear(const Tensor<Scalar>& input, const Tensor<Scalar>& weight,
       std::optional<std::reference_wrapper<const Tensor<Scalar>>> bias =
           std::nullopt) {
    if (weight.shape.size() != 2) {
        throw std::invalid_argument("linear weight must be a 2-D matrix");
    }
    Tensor<Scalar> output = matmul(input, transpose(weight, 0, 1));
    detail::add_bias(output, bias);
    return output;
}

/** Apply an int8 weight to input rows quantized on the fly (W8A8).
 *
 * Each input row is quantized with its own scale (see quantize_rows()),
 * multiplied by the int8 weight in int32, and each sum is rescaled as
 *   Scalar(sum) * (row_scale * weight.scales[column])
 * before the bias is added. The float steps happen in exactly that order,
 * each through safe_multiply() or safe_add().
 *
 * @tparam Scalar Floating-point tensor value type.
 * @param input Input rows shaped [rows, in_features].
 * @param weight int8 weight shaped [out_features, in_features] with one
 *        scale per output feature.
 * @param bias Optional reference to a bias shaped [out_features]; no copy is
 *        made.
 * @return Projected rows shaped [rows, out_features].
 * @throws std::invalid_argument If the shapes or value counts are
 * incompatible, the bias is not one value per output feature, or the input
 * holds an infinite or NaN value.
 * @throws std::overflow_error If a rescaled output cannot be represented.
 */
template <typename Scalar>
Tensor<Scalar>
linear(const Tensor<Scalar>& input, const QuantizedWeight<Scalar>& weight,
       std::optional<std::reference_wrapper<const Tensor<Scalar>>> bias =
           std::nullopt) {
    if (weight.values.shape.size() != 2 ||
        weight.scales.size() != weight.values.shape[0]) {
        throw std::invalid_argument(
            "linear int8 weight must be a 2-D matrix with one scale per row");
    }
    const QuantizedWeight<Scalar> rows = quantize_rows(input);
    const Tensor<std::int32_t> sums = matmul<std::int8_t, std::int32_t>(
        rows.values, transpose(weight.values, 0, 1));

    const std::size_t out_features = sums.shape[1];
    Tensor<Scalar> output{sums.shape, std::vector<Scalar>(sums.values.size())};
    for (std::size_t row = 0; row < sums.shape[0]; ++row) {
        for (std::size_t column = 0; column < out_features; ++column) {
            const std::size_t index = (row * out_features) + column;
            output.values[index] = safe_multiply(
                static_cast<Scalar>(sums.values[index]),
                safe_multiply(rows.scales[row], weight.scales[column]));
        }
    }
    detail::add_bias(output, bias);
    return output;
}

} // namespace inference_engine
