#pragma once

/** @file
 *  @brief Storage for typed inference tensors
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace inference_engine {

/**
 *
 * @tparam Scalar Type of each stored value.
 */
template <typename Scalar> struct Tensor {
    /// Tensor dimensions in axis order.
    std::vector<std::size_t> shape;
    /// Values in row-major order for two-dimensional tensors.
    std::vector<Scalar> values;
};

/** Check that a tensor's value count matches its declared shape.
 *
 * An empty shape describes a scalar holding one value.
 *
 * @tparam Scalar Tensor value type.
 * @param tensor Tensor to inspect.
 * @return Whether every dimension is nonzero, values.size() equals the
 * product of the shape dimensions, and that product fits in std::size_t.
 */
template <typename Scalar> bool valid_tensor(const Tensor<Scalar>& tensor) {
    std::size_t count = 1;
    for (const auto dimension : tensor.shape) {
        if (dimension == 0 ||
            count > std::numeric_limits<std::size_t>::max() / dimension) {
            return false;
        }
        count *= dimension;
    }
    return tensor.values.size() == count;
}

/** Check the storage and shape of one two-dimensional matrix.
 *
 * @tparam Scalar Tensor value type.
 * @param tensor Matrix to inspect.
 * @return Whether the matrix has valid nonempty row-major storage.
 */
template <typename Scalar> bool valid_matrix(const Tensor<Scalar>& tensor) {
    return tensor.shape.size() == 2 && valid_tensor(tensor);
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
template <typename Scalar>
Tensor<Scalar> matmul(const Tensor<Scalar>& left, const Tensor<Scalar>& right) {
    static_assert(std::is_arithmetic_v<Scalar> && !std::is_same_v<Scalar, bool>,
                  "matmul requires a non-bool arithmetic scalar");
    static_assert(!std::is_integral_v<Scalar> ||
                      sizeof(Scalar) <= sizeof(std::uint32_t),
                  "matmul supports integers up to 32 bits");

    if (!valid_matrix(left) || !valid_matrix(right) ||
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
                matrix_dot_product(left, right, row, column);
        }
    }
    return result;
}

} // namespace inference_engine
