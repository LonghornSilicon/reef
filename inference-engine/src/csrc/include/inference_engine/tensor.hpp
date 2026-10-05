#pragma once

/** @file
 *  @brief Storage for typed inference tensors and host matrix operators.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace inference_engine {

/** Host-side tensor with row-major values for two-dimensional operations.
 *
 * @tparam Scalar Type of each stored value.
 */
template <typename Scalar> struct Tensor {
    /// Tensor dimensions in axis order.
    std::vector<std::size_t> shape;
    /// Values in row-major order for two-dimensional tensors.
    std::vector<Scalar> values;
};

/** Internal helpers for the host matrix reference implementation. */
namespace detail {

/** Check the storage and shape of one two-dimensional matrix.
 *
 * @tparam Scalar Tensor value type.
 * @param tensor Matrix to inspect.
 * @return Whether the matrix has valid nonempty row-major storage.
 */
template <typename Scalar> bool valid_matrix(const Tensor<Scalar>& tensor) {
    return tensor.shape.size() == 2 && tensor.shape[0] > 0 &&
           tensor.shape[1] > 0 &&
           tensor.shape[0] <=
               std::numeric_limits<std::size_t>::max() / tensor.shape[1] &&
           tensor.values.size() == tensor.shape[0] * tensor.shape[1];
}

/** Compute one matrix output value with checked integer accumulation.
 *
 * Shapes and storage must already have been validated.
 *
 * @tparam Scalar Tensor value type.
 * @param left Left matrix.
 * @param right Right matrix.
 * @param row Output row.
 * @param column Output column.
 * @return One output value.
 * @throws std::overflow_error If an integer result cannot be represented.
 */
template <typename Scalar>
Scalar matrix_dot_product(const Tensor<Scalar>& left,
                          const Tensor<Scalar>& right, std::size_t row,
                          std::size_t column) {
    using Accumulator =
        std::conditional_t<std::is_floating_point_v<Scalar>, Scalar,
                           std::conditional_t<std::is_signed_v<Scalar>,
                                              std::int64_t, std::uint64_t>>;
    const auto inner = left.shape[1];
    const auto columns = right.shape[1];
    Accumulator sum{};
    for (std::size_t index = 0; index < inner; ++index) {
        const auto product =
            static_cast<Accumulator>(left.values[(row * inner) + index]) *
            static_cast<Accumulator>(right.values[(index * columns) + column]);
        if constexpr (std::is_integral_v<Scalar>) {
            if (product > 0 &&
                sum > std::numeric_limits<Accumulator>::max() - product) {
                throw std::overflow_error("matmul sum overflow");
            }
            if constexpr (std::is_signed_v<Scalar>) {
                if (product < 0 &&
                    sum < std::numeric_limits<Accumulator>::min() - product) {
                    throw std::overflow_error("matmul sum overflow");
                }
            }
        }
        sum += product;
    }
    if constexpr (std::is_integral_v<Scalar>) {
        if (sum <
                static_cast<Accumulator>(std::numeric_limits<Scalar>::min()) ||
            sum >
                static_cast<Accumulator>(std::numeric_limits<Scalar>::max())) {
            throw std::overflow_error("matmul result overflow");
        }
    }
    return static_cast<Scalar>(sum);
}

} // namespace detail

/** Multiply two row-major, two-dimensional arithmetic tensors.
 *
 * @tparam Scalar Arithmetic value type. Integers up to 32 bits are supported,
 * and integer results must fit in Scalar.
 * @param left Left matrix.
 * @param right Right matrix.
 * @return Product with shape left.rows by right.columns.
 * @throws std::invalid_argument If shapes or value counts are incompatible.
 * @throws std::overflow_error If an integer result cannot be represented.
 */
// @ MLP team
// TODO: Agree on target accumulation and quantization for inference kernels;
// Attention will also use this host reference primitive.
template <typename Scalar>
Tensor<Scalar> matmul(const Tensor<Scalar>& left, const Tensor<Scalar>& right) {
    static_assert(std::is_arithmetic_v<Scalar> && !std::is_same_v<Scalar, bool>,
                  "matmul requires a non-bool arithmetic scalar");
    static_assert(!std::is_integral_v<Scalar> ||
                      sizeof(Scalar) <= sizeof(std::uint32_t),
                  "matmul supports integers up to 32 bits");

    if (!detail::valid_matrix(left) || !detail::valid_matrix(right) ||
        left.shape[1] != right.shape[0] ||
        left.shape[0] >
            std::numeric_limits<std::size_t>::max() / right.shape[1]) {
        throw std::invalid_argument("matmul requires compatible 2-D matrices");
    }

    const auto rows = left.shape[0];
    const auto columns = right.shape[1];
    Tensor<Scalar> result{{rows, columns},
                          std::vector<Scalar>(rows * columns, Scalar{})};
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            result.values[(row * columns) + column] =
                detail::matrix_dot_product(left, right, row, column);
        }
    }
    return result;
}

} // namespace inference_engine
