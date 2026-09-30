#pragma once

#include "inference_engine/embedding.hpp"

namespace inference_engine {

// TODO: Define a fixed-weight model and per-layer cache after the three teams
// agree on Tensor, weight, and token-ID contracts.
struct ModelWeights;
struct ModelState;

// TODO (integration owner to be assigned): Compose embedding, attention, and
// MLP into one inference pass after the three team interfaces are agreed.
// Keep model execution separate from a future standalone main().
Tensor infer(const TokenIds& ids, const ModelWeights& weights,
             ModelState& state);

}  // namespace inference_engine
