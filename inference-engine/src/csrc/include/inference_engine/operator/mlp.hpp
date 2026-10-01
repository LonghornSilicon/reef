#pragma once

#include "inference_engine/tensor.hpp"

namespace inference_engine {

// TODO: Define the two linear layers' weights and biases once the shared
// Tensor representation is agreed.
struct MlpWeights;

// @ MLP team
// TODO: Keep the two layer weights together and make forward usable by model
// tests and experiments. Match GPTNeoMLP in workloads.
class MLP {
  public:
    explicit MLP(const MlpWeights& weights);

    // TODO: Compose linear -> GELU -> linear. Check intermediate width and
    // output shape against a fixed reference case.
    [[nodiscard]] Tensor forward(const Tensor& input) const;

  private:
    const MlpWeights& weights_;
};

} // namespace inference_engine
