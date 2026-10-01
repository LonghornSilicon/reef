#pragma once

/** @file
 *  @brief Elementwise activations and dimension-aware softmax.
 */

#include "inference_engine/tensor.hpp"

#include <cstddef>

namespace inference_engine {

/** Apply the tanh-approximation GELU elementwise.
 *
 * @param input Input values.
 * @return Tensor with the same shape after activation.
 */
// @ MLP team
// TODO: Implement and set a numeric tolerance for host and Coral comparisons.
Tensor<float> gelu(const Tensor<float>& input);

/** Apply softmax along one tensor dimension, including axis zero for 1-D data.
 *
 * @param input Input values.
 * @param dim Zero-based dimension to normalize.
 * @return Tensor with the same shape and normalized values along @p dim.
 */
// @ Attention team
// TODO: Implement a numerically stable operation; define masked-row behavior
// and accumulation precision.
Tensor<float> softmax(const Tensor<float>& input, std::size_t dim);

} // namespace inference_engine
