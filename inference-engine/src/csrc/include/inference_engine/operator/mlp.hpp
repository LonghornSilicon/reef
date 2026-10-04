#pragma once

/** @file
 *  @brief Feed-forward operator interface.
 */

#include "inference_engine/tensor.hpp"

namespace inference_engine {

/** Weights and biases for the two linear layers. */
// TODO: Define the representation after the shared Tensor contract is agreed.
/** Parameters for the GPT-Neo feed-forward block. */
struct MlpWeights {
    /// First projection weights: [intermediate_size, hidden_size].
    Tensor<float> c_fc_weight;
    /// First projection bias: [intermediate_size].
    Tensor<float> c_fc_bias;
    /// Second projection weights: [hidden_size, intermediate_size].
    Tensor<float> c_proj_weight;
    /// Second projection bias: [hidden_size].
    Tensor<float> c_proj_bias;
};

/** Apply the model's two-layer feed-forward block. */
// @ MLP team
// TODO: Keep the two layer weights together and make forward usable by model
// tests and experiments. Match GPTNeoMLP in workloads.
class MLP {
  public:
    /** Bind externally owned layer parameters.
     *
     * @param weights Immutable parameters for both layers.
     */
    explicit MLP(const MlpWeights& weights);

    /** Run the two linear layers and activation.
     *
     * @param input Input tensor.
     * @return Output tensor after the feed-forward block.
     */
    // TODO: Compose linear -> GELU -> linear. Check intermediate width and
    // output shape against a fixed reference case.
    [[nodiscard]] Tensor<float> forward(const Tensor<float>& input) const;

  private:
    const MlpWeights& weights_;
};

} // namespace inference_engine
