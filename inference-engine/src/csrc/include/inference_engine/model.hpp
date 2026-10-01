#pragma once

/** @file
 *  @brief GPT-Neo inference model interface.
 */

#include "inference_engine/operator/embedding.hpp"

namespace inference_engine {

/** Model parameters, to be defined when operator contracts are agreed. */
struct ModelWeights;
/** Mutable runtime state, including the per-layer decode cache. */
struct ModelState;

/** Compose embeddings, attention, and MLP blocks for GPT-Neo inference. */
// TODO (integration owner to be assigned): Keep model weights and runtime
// state inside the object so tests and experiments can call forward repeatedly.
class GPTNeo {
  public:
    /** Bind externally owned weights and runtime state.
     *
     * @param weights Immutable model parameters.
     * @param state Mutable inference state.
     */
    GPTNeo(const ModelWeights& weights, ModelState& state);

    /** Run one prefill or decode step.
     *
     * @param ids Input token IDs.
     * @return Output tensor for the supplied tokens.
     */
    // TODO: Compose embedding, attention, and MLP after their interfaces agree.
    // Keep model execution separate from a future standalone main().
    Tensor<float> forward(const TokenIds& ids);

  private:
    const ModelWeights& weights_;
    ModelState& state_;
};

} // namespace inference_engine
