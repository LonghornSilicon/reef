#pragma once

/** @file
 *  @brief Storage for typed inference tensors and their core operators.
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace inference_engine {

/** Host-side tensor with row-major values for two-dimensional operations.
 *
 * TODO: Agree across teams on shape, scalar types, and ownership for the
 * target backend.
 *
 * @tparam Scalar Type of each stored value.
 */
template <typename Scalar> struct Tensor {
    /// Tensor dimensions in axis order.
    std::vector<std::size_t> shape;
    /// Values in row-major order for two-dimensional tensors.
    std::vector<Scalar> values;
};

namespace detail {

/** Count the values a shape holds.
 *
 * @param shape Tensor dimensions.
 * @return Product of @p shape, or 1 for a scalar.
 */
inline std::size_t element_count(const std::vector<std::size_t>& shape) {
    std::size_t count = 1;
    for (const std::size_t dim : shape) {
        count *= dim;
    }
    return count;
}

/** Compute row-major strides for a shape.
 *
 * @param shape Tensor dimensions.
 * @return Values to skip per step along each axis.
 */
inline std::vector<std::size_t>
row_major_strides(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t axis = shape.size(); axis > 1; --axis) {
        strides[axis - 2] = strides[axis - 1] * shape[axis - 1];
    }
    return strides;
}

} // namespace detail

/** Swap two dimensions, reordering values to match the new row-major layout.
 *
 * @tparam Scalar Type of each stored value.
 * @param tensor Input tensor.
 * @param dim0 First dimension to swap.
 * @param dim1 Second dimension to swap.
 * @return Tensor with @p dim0 and @p dim1 exchanged.
 * @throws std::invalid_argument If a dimension is out of range or the shape
 *         does not match the number of values.
 */
template <typename Scalar>
Tensor<Scalar> transpose(const Tensor<Scalar>& tensor, std::size_t dim0,
                         std::size_t dim1) {
    const std::size_t rank = tensor.shape.size();
    if (dim0 >= rank || dim1 >= rank) {
        throw std::invalid_argument(
            "transpose dims must be less than the rank");
    }
    if (detail::element_count(tensor.shape) != tensor.values.size()) {
        throw std::invalid_argument("transpose shape does not match values");
    }

    std::vector<std::size_t> out_shape = tensor.shape;
    out_shape[dim0] = tensor.shape[dim1];
    out_shape[dim1] = tensor.shape[dim0];

    const std::vector<std::size_t> in_strides =
        detail::row_major_strides(tensor.shape);
    const std::vector<std::size_t> out_strides =
        detail::row_major_strides(out_shape);

    Tensor<Scalar> result{out_shape, std::vector<Scalar>(tensor.values.size())};
    for (std::size_t in = 0; in < tensor.values.size(); ++in) {
        std::size_t out = 0;
        for (std::size_t axis = 0; axis < rank; ++axis) {
            const std::size_t coord =
                (in / in_strides[axis]) % tensor.shape[axis];
            std::size_t out_axis = axis;
            if (axis == dim0) {
                out_axis = dim1;
            } else if (axis == dim1) {
                out_axis = dim0;
            }
            out += coord * out_strides[out_axis];
        }
        result.values[out] = tensor.values[in];
    }
    return result;
}

/** Reinterpret values with a new shape without reordering them.
 *
 * @tparam Scalar Type of each stored value.
 * @param tensor Input tensor.
 * @param shape New dimensions; their product must equal the value count.
 * @return Tensor with the same values and @p shape.
 * @throws std::invalid_argument If @p shape holds a different number of
 *         values.
 */
template <typename Scalar>
Tensor<Scalar> reshape(const Tensor<Scalar>& tensor,
                       const std::vector<std::size_t>& shape) {
    if (detail::element_count(shape) != tensor.values.size()) {
        throw std::invalid_argument("reshape shape does not match values");
    }
    return Tensor<Scalar>{shape, tensor.values};
}

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

/** @cond INTERNAL */
extern template Tensor<float> matmul(const Tensor<float>&,
                                     const Tensor<float>&);
extern template Tensor<std::int8_t> matmul(const Tensor<std::int8_t>&,
                                           const Tensor<std::int8_t>&);
/** @endcond */

/** Apply a weight matrix and an optional bias to input rows.
 *
 * @tparam Scalar Tensor value type.
 * @param input Input rows.
 * @param weight Matrix of output weights.
 * @param bias Optional reference to a bias tensor; no copy is made.
 * @return Transformed rows.
 */
// @ MLP team
// TODO: Apply x @ weight.T + optional bias with the same weight layout as the
// workloads reference.
template <typename Scalar>
Tensor<Scalar>
linear(const Tensor<Scalar>& input, const Tensor<Scalar>& weight,
       std::optional<std::reference_wrapper<const Tensor<Scalar>>> bias =
           std::nullopt);

} // namespace inference_engine
