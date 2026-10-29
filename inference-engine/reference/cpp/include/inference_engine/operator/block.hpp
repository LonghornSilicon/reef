#pragma once

/** @file
 *  @brief One GPT-Neo transformer layer with its residual connections.
 */

#include "inference_engine/operator/attention.hpp"
#include "inference_engine/operator/layernorm.hpp"
#include "inference_engine/operator/mlp.hpp"
#include "inference_engine/tensor.hpp"
#include "inference_engine/util.hpp"
#include "inference_engine/weights.hpp"

#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>

namespace inference_engine {

/** Add a sublayer's output to its input, elementwise.
 *
 * @tparam Scalar Tensor value type.
 * @param input Residual stream.
 * @param sublayer Output of attention or the MLP, shaped like @p input.
 * @return input + sublayer, each element added in that operand order.
 * @throws std::invalid_argument If the shapes differ.
 * @throws std::overflow_error If a sum cannot be represented.
 */
template <typename Scalar>
Tensor<Scalar> residual_add(const Tensor<Scalar>& input,
                            const Tensor<Scalar>& sublayer) {
    if (input.shape != sublayer.shape || !valid_tensor(input) ||
        !valid_tensor(sublayer)) {
        throw std::invalid_argument("residual_add requires equal shapes");
    }
    Tensor<Scalar> output = input;
    for (std::size_t index = 0; index < output.values.size(); ++index) {
        output.values[index] =
            safe_add(output.values[index], sublayer.values[index]);
    }
    return output;
}

/** One pre-norm GPT-Neo layer that keeps its own key and value cache.
 *
 * forward computes
 *   x = x + attention(layernorm(x, ln_1))
 *   x = x + mlp(layernorm(x, ln_2))
 * with global or sliding-window attention depending on the layer index.
 * Every step is row-independent apart from attention, which reads the
 * cache, so a prompt gives the same bits whether it arrives at once or one
 * token at a time.
 *
 * @tparam Scalar Floating-point tensor value type.
 */
template <typename Scalar> class Block {
  public:
    /** Bind one layer's weights and create its empty cache.
     *
     * @param weights Layer parameters; must outlive the block.
     * @param config Model hyperparameters.
     * @param layer Zero-based layer index, which picks global or local
     *        attention.
     */
    Block(const LayerWeights<Scalar>& weights, const GptNeoConfig& config,
          std::size_t layer)
        : weights_(weights), eps_(static_cast<Scalar>(config.layer_norm_eps)),
          mlp_(weights.mlp) {
        if (config.is_local(layer)) {
            attention_ = std::make_unique<LocalAttention<Scalar>>(
                weights.attention, cache_, config.window_size);
        } else {
            attention_ = std::make_unique<GlobalAttention<Scalar>>(
                weights.attention, cache_);
        }
    }

    /// The attention object refers to cache_, so a block never moves.
    Block(const Block&) = delete;
    /// The attention object refers to cache_, so a block never moves.
    Block& operator=(const Block&) = delete;
    /// The attention object refers to cache_, so a block never moves.
    Block(Block&&) = delete;
    /// The attention object refers to cache_, so a block never moves.
    Block& operator=(Block&&) = delete;
    /// Destroy the block and its cache.
    ~Block() = default;

    /** Run the layer on the next positions of the sequence.
     *
     * @param input Rows [length, hidden] for the positions after those
     *        already cached.
     * @return Rows [length, hidden].
     */
    Tensor<Scalar> forward(const Tensor<Scalar>& input) {
        const Tensor<Scalar> attended =
            residual_add(input, attention_->forward(layernorm<Scalar>(
                                    input, weights_.ln_1_weight,
                                    std::cref(weights_.ln_1_bias), eps_)));
        Tensor<Scalar> output =
            residual_add(attended, mlp_.forward(layernorm<Scalar>(
                                       attended, weights_.ln_2_weight,
                                       std::cref(weights_.ln_2_bias), eps_)));
        assert(output.shape == input.shape);
        return output;
    }

    /** Forget every cached position, ready for a new sequence. */
    void clear_cache() { cache_ = KvCache<Scalar>{}; }

  private:
    const LayerWeights<Scalar>& weights_;
    Scalar eps_;
    KvCache<Scalar> cache_;
    std::unique_ptr<Attention<Scalar>> attention_;
    MLP<Scalar> mlp_;
};

} // namespace inference_engine
