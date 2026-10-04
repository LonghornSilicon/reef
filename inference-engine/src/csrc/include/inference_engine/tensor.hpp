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

} // namespace inference_engine
