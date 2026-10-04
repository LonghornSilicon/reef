#pragma once

/** @file
 *  @brief Storage for typed inference tensors.
 */

#include <cstddef>
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

/** Swap two dimensions, reordering values to match the new row-major layout.
 *
 * @param tensor Input tensor.
 * @param dim0 First dimension to swap.
 * @param dim1 Second dimension to swap.
 * @return Tensor with @p dim0 and @p dim1 exchanged.
 * @throws std::invalid_argument If a dimension is out of range or the shape
 *         does not match the number of values.
 */
Tensor<float> transpose(const Tensor<float>& tensor, std::size_t dim0,
                        std::size_t dim1);

/** Reinterpret values with a new shape without reordering them.
 *
 * @param tensor Input tensor.
 * @param shape New dimensions; their product must equal the value count.
 * @return Tensor with the same values and @p shape.
 * @throws std::invalid_argument If @p shape holds a different number of
 *         values.
 */
Tensor<float> reshape(const Tensor<float>& tensor,
                      const std::vector<std::size_t>& shape);

} // namespace inference_engine
