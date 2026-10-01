#pragma once

/** @file
 *  @brief Floating-point matrix operators for the first host slice.
 */

#include "inference_engine/tensor.hpp"

#include <functional>
#include <optional>

namespace inference_engine {

/** Multiply two row-major, two-dimensional float tensors.
 *
 * @param left Left matrix.
 * @param right Right matrix.
 * @return Product with shape left.rows by right.columns.
 * @throws std::invalid_argument If shapes or value counts are incompatible.
 */
// @ MLP team
// TODO: Agree on the target accumulation type and larger shapes; Attention
// will also use this primitive.
Tensor<float> matmul(const Tensor<float>& left, const Tensor<float>& right);

/** Apply a weight matrix and an optional bias to input rows.
 *
 * @param input Input rows.
 * @param weight Matrix of output weights.
 * @param bias Optional reference to a bias tensor; no copy is made.
 * @return Transformed rows.
 */
// @ MLP team
// TODO: Apply x @ weight.T + optional bias with the same weight layout as the
// workloads reference.
Tensor<float> linear(const Tensor<float>& input, const Tensor<float>& weight,
                     std::optional<std::reference_wrapper<const Tensor<float>>>
                         bias = std::nullopt);

} // namespace inference_engine
