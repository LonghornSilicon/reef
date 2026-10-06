#pragma once

/** @file
 *  @brief Storage for typed inference tensors
 */

#include <cmath>
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

/** Multiply two accumulator values, rejecting integer overflow.
 *
 * @tparam Accumulator Arithmetic type the product is computed in.
 * @param left Left factor.
 * @param right Right factor.
 * @return left * right.
 * @throws std::overflow_error If an integer product cannot be represented.
 */
template <typename Accumulator>
Accumulator checked_multiply(Accumulator left, Accumulator right) {
    if constexpr (std::is_integral_v<Accumulator>) {
        constexpr auto kMax = std::numeric_limits<Accumulator>::max();
        constexpr auto kMin = std::numeric_limits<Accumulator>::min();
        bool overflow = false;
        if constexpr (std::is_signed_v<Accumulator>) {
            if (left > 0) {
                overflow =
                    right > 0 ? left > kMax / right : right < kMin / left;
            } else if (left < 0) {
                overflow = right > 0 ? left < kMin / right
                                     : right < 0 && right < kMax / left;
            }
        } else {
            overflow = left != 0 && right > kMax / left;
        }
        if (overflow) {
            throw std::overflow_error("matmul product overflow");
        }
    }
    return static_cast<Accumulator>(left * right);
}

/** Compute one matrix output value with checked accumulation.
 *
 * Shapes and storage must already have been validated.
 *
 * @tparam Scalar Input tensor value type.
 * @tparam Accumulator Type the products are summed and returned in.
 * @param left Left matrix.
 * @param right Right matrix.
 * @param row Output row.
 * @param column Output column.
 * @return One output value.
 * @throws std::overflow_error If an integer product or sum cannot be
 * represented or a floating-point result is infinite.
 */
template <typename Scalar, typename Accumulator = Scalar>
Accumulator matrix_dot_product(const Tensor<Scalar>& left,
                               const Tensor<Scalar>& right, std::size_t row,
                               std::size_t column) {
    const auto inner = left.shape[1];
    const auto columns = right.shape[1];
    Accumulator sum{};
    for (std::size_t index = 0; index < inner; ++index) {
        const auto product = checked_multiply(
            static_cast<Accumulator>(left.values[(row * inner) + index]),
            static_cast<Accumulator>(right.values[(index * columns) + column]));
        if constexpr (std::is_integral_v<Accumulator>) {
            if (product > 0 &&
                sum > std::numeric_limits<Accumulator>::max() - product) {
                throw std::overflow_error("matmul sum overflow");
            }
            if constexpr (std::is_signed_v<Accumulator>) {
                if (product < 0 &&
                    sum < std::numeric_limits<Accumulator>::min() - product) {
                    throw std::overflow_error("matmul sum overflow");
                }
            }
        }
        sum += product;
    }
    if constexpr (std::is_floating_point_v<Accumulator>) {
        if (std::isinf(sum)) {
            throw std::overflow_error("matmul result overflow");
        }
    }
    return sum;
}

/** Multiply two row-major, two-dimensional arithmetic tensors.
 *
 * Mirrors hardware such as tensor cores that take narrow inputs and produce
 * wider outputs, e.g. matmul<std::int8_t, std::int32_t>.
 *
 * @tparam Scalar Input tensor value type.
 * @tparam Accumulator Type the products are summed and returned in. It must
 * represent every Scalar value; it defaults to Scalar.
 * @param left Left matrix.
 * @param right Right matrix.
 * @return Product with shape left.rows by right.columns.
 * @throws std::invalid_argument If shapes or value counts are incompatible.
 * @throws std::overflow_error If an integer product or sum cannot be
 * represented or a floating-point result is infinite.
 */
template <typename Scalar, typename Accumulator = Scalar>
Tensor<Accumulator> matmul(const Tensor<Scalar>& left,
                           const Tensor<Scalar>& right) {
    static_assert(std::is_arithmetic_v<Scalar> && !std::is_same_v<Scalar, bool>,
                  "matmul requires a non-bool arithmetic scalar");
    static_assert(std::is_arithmetic_v<Accumulator> &&
                      !std::is_same_v<Accumulator, bool>,
                  "matmul requires a non-bool arithmetic accumulator");
    static_assert(
        std::is_floating_point_v<Accumulator> ||
            (std::is_integral_v<Scalar> &&
             (std::is_signed_v<Accumulator> || !std::is_signed_v<Scalar>)),
        "matmul accumulator must represent every scalar value");
    static_assert(std::numeric_limits<Accumulator>::digits >=
                      std::numeric_limits<Scalar>::digits,
                  "matmul accumulator must represent every scalar value");

    if (!valid_matrix(left) || !valid_matrix(right) ||
        left.shape[1] != right.shape[0] ||
        left.shape[0] >
            std::numeric_limits<std::size_t>::max() / right.shape[1]) {
        throw std::invalid_argument("matmul requires compatible 2-D matrices");
    }

    const auto rows = left.shape[0];
    const auto columns = right.shape[1];
    Tensor<Accumulator> result{
        {rows, columns},
        std::vector<Accumulator>(rows * columns, Accumulator{})};
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            result.values[(row * columns) + column] =
                matrix_dot_product<Scalar, Accumulator>(left, right, row,
                                                        column);
        }
    }
    return result;
}

} // namespace inference_engine
