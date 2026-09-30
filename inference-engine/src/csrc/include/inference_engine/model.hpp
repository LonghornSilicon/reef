#pragma once

#include "inference_engine/operator/embedding.hpp"

namespace inference_engine {

// TODO: Define a fixed-weight model and per-layer cache after the three teams
// agree on Tensor, weight, and token-ID contracts.
struct ModelWeights;
struct ModelState;

// TODO (integration owner to be assigned): Keep model weights and runtime
// state inside the object so tests and experiments can call forward repeatedly.
class GPTNeo {
  public:
    GPTNeo(const ModelWeights& weights, ModelState& state);

    // TODO: Compose embedding, attention, and MLP after their interfaces agree.
    // Keep model execution separate from a future standalone main().
    Tensor forward(const TokenIds& ids);

  private:
    const ModelWeights& weights_;
    ModelState& state_;
};

} // namespace inference_engine
