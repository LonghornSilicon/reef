#pragma once

/** @file
 *  @brief Storage for typed inference tensors.
 */

#include <cstddef>
#include <stdexcept>
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

} // namespace inference_engine
