#pragma once

/** @file
 *  @brief Feed-forward operator interface.
 */

#include "inference_engine/tensor.hpp"

namespace inference_engine {

/** Weights and biases for the two linear layers.
 *
 * The feed-forward block is:
 *   W_2 GELU(W_1 x + b_1) + b_2
 */
struct MlpWeights {
    /** Input feature width; also the output feature width of the projection. */
    std::size_t input_size{};
    /** Hidden/expanded width of the first layer. */
    std::size_t hidden_size{};
    /** First linear layer weight matrix, stored as [hidden_size, input_size].
     */
    Tensor<float> W_1{};
    /** First linear layer bias values, shaped [hidden_size]. */
    Tensor<float> b_1{};
    /** Second linear layer weight matrix, stored as [input_size, hidden_size].
     */
    Tensor<float> W_2{};
    /** Second linear layer bias values, shaped [input_size]. */
    Tensor<float> b_2{};
};

/** Apply the model's two-layer feed-forward block. */
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
    [[nodiscard]] Tensor<float> forward(const Tensor<float>& input) const;

  private:
    const MlpWeights& weights_;
};

} // namespace inference_engine
