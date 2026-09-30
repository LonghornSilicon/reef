#pragma once

#include "inference_engine/tensor.hpp"

namespace inference_engine {

// @ MLP team
// Multiply 2-D row-major float tensors. Throws std::invalid_argument for bad
// shapes or storage. TODO: Agree on target accumulation type and larger shapes;
// Attention will also use this primitive.
Tensor matmul(const Tensor& left, const Tensor& right);

// @ MLP team
// TODO: Apply x @ weight.T + optional bias with the same weight layout as the
// workloads reference. Decide how an absent bias is represented.
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor* bias);

}  // namespace inference_engine
