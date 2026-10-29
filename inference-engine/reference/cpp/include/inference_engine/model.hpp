#pragma once

/** @file
 *  @brief GPT-Neo inference: forward passes and greedy generation.
 */

#include "inference_engine/operator/block.hpp"
#include "inference_engine/operator/embedding.hpp"
#include "inference_engine/operator/layernorm.hpp"
#include "inference_engine/operator/linear.hpp"
#include "inference_engine/tensor.hpp"
#include "inference_engine/weights.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace inference_engine {

/** Index of the largest value, the first one on ties.
 *
 * @tparam Scalar Tensor value type.
 * @param row Values to search, such as one row of logits; any shape.
 * @return Position of the first maximum in row-major order.
 * @throws std::invalid_argument If @p row holds no values.
 */
template <typename Scalar> std::size_t argmax(const Tensor<Scalar>& row) {
    if (row.values.empty()) {
        throw std::invalid_argument("argmax requires at least one value");
    }
    std::size_t best = 0;
    for (std::size_t index = 1; index < row.values.size(); ++index) {
        if (row.values[index] > row.values[best]) {
            best = index;
        }
    }
    return best;
}

/** GPT-Neo language model over externally owned weights.
 *
 * Each forward call continues the sequence: it embeds the new tokens at the
 * next positions,
 *   x = wte[id] + wpe[position]
 * runs every layer (see Block), applies the final LayerNorm to the last row
 * and returns that row's logits, x @ wte.T with no bias. The model owns the
 * per-layer key and value caches. Weights carrying W8A8 copies (see
 * quantize_w8a8()) run their projections through int8.
 *
 * @tparam Scalar Floating-point tensor value type.
 */
template <typename Scalar> class GPTNeo {
  public:
    /** Bind weights and start an empty sequence.
     *
     * @param weights Model parameters; must outlive the model.
     * @throws std::invalid_argument If the weights do not match their config.
     */
    explicit GPTNeo(const ModelWeights<Scalar>& weights) : weights_(weights) {
        validate_config(weights.config);
        if (weights.layers.size() != weights.config.num_layers) {
            throw std::invalid_argument(
                "GPT-Neo weights must hold one entry per layer");
        }
        for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
            blocks_.push_back(std::make_unique<Block<Scalar>>(
                weights.layers[layer], weights.config, layer));
        }
    }

    /// The model keeps a reference, so temporary weights would dangle.
    explicit GPTNeo(ModelWeights<Scalar>&&) = delete;

    /** Append tokens to the sequence and score the next one.
     *
     * @param ids Token IDs for the next positions; at least one.
     * @return Logits [1, vocab] for the token after the last of @p ids.
     * @throws std::invalid_argument If @p ids is empty, holds an unknown ID,
     * or runs past the last position.
     */
    Tensor<Scalar> forward(const TokenIds& ids) {
        if (ids.empty()) {
            throw std::invalid_argument("GPT-Neo forward requires tokens");
        }
        Tensor<Scalar> x =
            add_position_embeddings(lookup_token_embeddings(ids, weights_.wte),
                                    weights_.wpe, position_);
        for (const auto& block : blocks_) {
            x = block->forward(x);
        }
        position_ += ids.size();

        // LayerNorm and the output projection treat rows independently, so
        // only the last row is needed.
        const std::size_t hidden = weights_.config.hidden_size;
        assert(x.shape == (std::vector<std::size_t>{ids.size(), hidden}));
        const std::size_t last = (ids.size() - 1) * hidden;
        Tensor<Scalar> last_row{{1, hidden}, std::vector<Scalar>(hidden)};
        for (std::size_t column = 0; column < hidden; ++column) {
            last_row.values.at(column) = x.values.at(last + column);
        }
        return linear<Scalar>(
            layernorm<Scalar>(
                last_row, weights_.ln_f_weight, std::cref(weights_.ln_f_bias),
                static_cast<Scalar>(weights_.config.layer_norm_eps)),
            weights_.wte);
    }

    /** Continue a prompt greedily from an empty sequence.
     *
     * Resets the model, feeds the whole prompt in one forward call, then
     * repeatedly appends the highest-scoring token (the first on ties).
     *
     * @param prompt Token IDs to start from; at least one.
     * @param new_tokens Number of tokens to generate.
     * @return The generated tokens, not including the prompt.
     * @throws std::invalid_argument If @p prompt is empty, holds an unknown
     * ID, or the sequence runs past the last position.
     */
    TokenIds generate(const TokenIds& prompt, std::size_t new_tokens) {
        if (prompt.empty()) {
            throw std::invalid_argument("GPT-Neo generate requires a prompt");
        }
        reset();
        TokenIds generated;
        if (new_tokens == 0) {
            return generated;
        }
        Tensor<Scalar> logits = forward(prompt);
        while (true) {
            generated.push_back(static_cast<std::uint32_t>(argmax(logits)));
            if (generated.size() == new_tokens) {
                return generated;
            }
            logits = forward({generated.back()});
        }
    }

    /** Clear every cache and return to position zero. */
    void reset() {
        for (const auto& block : blocks_) {
            block->clear_cache();
        }
        position_ = 0;
    }

  private:
    const ModelWeights<Scalar>& weights_;
    std::vector<std::unique_ptr<Block<Scalar>>> blocks_;
    std::size_t position_ = 0;
};

} // namespace inference_engine
