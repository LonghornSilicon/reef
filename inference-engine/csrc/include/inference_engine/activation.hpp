#pragma once

#include "inference_engine/tensor.hpp"

namespace inference_engine {

// @ MLP team
// TODO: Implement the workloads model's tanh-approximation GELU and set a
// numeric tolerance for host and Coral comparisons.
Tensor gelu(const Tensor& input);

// @ Attention team
// TODO: Compute numerically stable softmax over the last dimension. Define
// behavior for masked rows and the chosen accumulation precision.
Tensor softmax_last_dim(const Tensor& input);

} // namespace inference_engine
