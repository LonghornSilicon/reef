#pragma once

#include "inference_engine/tensor.hpp"

namespace inference_engine {

// TODO: Define the two linear layers' weights and biases once the shared
// Tensor representation is agreed.
struct MlpWeights;

// @ MLP team
// TODO: Compose linear -> GELU -> linear to match GPTNeoMLP in workloads.
// Check intermediate width and output shape against a fixed reference case.
Tensor mlp_forward(const Tensor& input, const MlpWeights& weights);

} // namespace inference_engine
