#pragma once

/** @file
 *  @brief GPT-Neo split across chips with tensor parallelism, rank by rank.
 */

#include "dist_inference/shard.hpp"

#include "inference_engine/model.hpp"
#include "inference_engine/operator/activation.hpp"
#include "inference_engine/operator/attention.hpp"
#include "inference_engine/operator/block.hpp"
#include "inference_engine/operator/embedding.hpp"
#include "inference_engine/operator/layernorm.hpp"
#include "inference_engine/operator/linear.hpp"
#include "inference_engine/tensor.hpp"
#include "inference_engine/weights.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dist_inference {

using inference_engine::ModelWeights;
using inference_engine::TokenIds;

/** How the two projections that read every head or the whole MLP width are
 * split. */
enum class TpScheme {
    /// Attention output and c_proj split by input rows; the host sums the
    /// partials. Two syncs per layer; differs from one chip by rounding.
    kMegatron,
    /// Every projection split by output columns; the host only concatenates.
    /// Four syncs per layer; the same bits as one chip.
    kColumnOnly,
};

/** One point where every rank stops and the host combines their outputs. */
struct Sync {
    /// What the host does with the ranks' outputs.
    enum class Kind {
        kGather, ///< Concatenate the ranks' slices.
        kSum,    ///< Add the ranks' full-width partials in rank order.
    };
    Kind kind{};         ///< Gather or sum.
    std::size_t rows{};  ///< Positions in this forward call.
    std::size_t width{}; ///< Values per row in the combined result.
};

/** GPT-Neo computed as @p ranks chips running tensor parallelism.
 *
 * Every rank holds a contiguous share of the heads and of the MLP width.
 * Per layer, each rank:
 *   - runs LayerNorm on the whole residual stream (all ranks hold it);
 *   - projects query, key and value for its heads only, keeps those heads'
 *     key and value cache, and attends them;
 *   - computes its columns of c_fc and applies GELU to them.
 * The attention output projection and c_proj are split as @p scheme says,
 * and the host combines the ranks' outputs at each Sync. The host also
 * embeds the tokens, and the tied lm_head is split by vocabulary rows and
 * gathered.
 *
 * kColumnOnly gives the same bits as inference_engine::GPTNeo for any rank
 * count. kMegatron gives the same bits with one rank, and differs by rounding
 * only with more. Both are built from the reference engine's operators, so a
 * change to the engine's arithmetic reaches this model too.
 *
 * @tparam Scalar Floating-point tensor value type.
 */
template <typename Scalar> class TpGptNeo {
  public:
    /** Bind weights and start an empty sequence.
     *
     * @param weights Model parameters; must outlive the model. W8A8 copies
     *        (quantize_w8a8()) run the projections through int8.
     * @param ranks Chips to split across; at least one.
     * @param scheme How the attention output and c_proj are split.
     * @throws std::invalid_argument If the weights do not match their config,
     * or @p ranks is zero or does not divide the heads and the MLP width.
     */
    TpGptNeo(const ModelWeights<Scalar>& weights, std::size_t ranks,
             TpScheme scheme = TpScheme::kMegatron)
        : weights_(weights), ranks_(ranks), scheme_(scheme) {
        const inference_engine::GptNeoConfig& config = weights.config;
        inference_engine::validate_config(config);
        if (weights.layers.size() != config.num_layers) {
            throw std::invalid_argument(
                "GPT-Neo weights must hold one entry per layer");
        }
        if (ranks == 0 || config.num_heads % ranks != 0 ||
            config.inner_size % ranks != 0) {
            throw std::invalid_argument(
                "tensor-parallel ranks must divide the heads and the MLP "
                "width");
        }
        reset();
    }

    /// The model keeps a reference, so temporary weights would dangle.
    TpGptNeo(ModelWeights<Scalar>&&, std::size_t,
             TpScheme = TpScheme::kMegatron) = delete;

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
        syncs_.clear();
        // The host embeds: wte is split by vocabulary rows for the lm_head,
        // so no rank holds the whole table.
        Tensor<Scalar> x = inference_engine::add_position_embeddings(
            inference_engine::lookup_token_embeddings(ids, weights_.wte),
            weights_.wpe, position_);
        for (std::size_t layer = 0; layer < weights_.layers.size(); ++layer) {
            x = layer_forward(layer, x);
        }
        position_ += ids.size();

        // LayerNorm and the lm_head treat rows independently, so only the
        // last row is needed.
        const Range last{ids.size() - 1, 1};
        const Tensor<Scalar> normed = inference_engine::layernorm<Scalar>(
            slice_rows(x, last), weights_.ln_f_weight,
            std::cref(weights_.ln_f_bias), eps());
        std::vector<Tensor<Scalar>> logits;
        for (std::size_t rank = 0; rank < ranks_; ++rank) {
            const Range vocab =
                shard_range(weights_.config.vocab_size, ranks_, rank);
            logits.push_back(inference_engine::linear<Scalar>(
                normed, slice_rows(weights_.wte, vocab)));
        }
        record(Sync::Kind::kGather, 1, weights_.config.vocab_size);
        return concat_columns(logits);
    }

    /** Continue a prompt greedily from an empty sequence.
     *
     * Matches inference_engine::GPTNeo::generate(): the whole prompt in one
     * forward call, then the highest-scoring token (the first on ties).
     *
     * @param prompt Token IDs to start from; at least one.
     * @param new_tokens Number of tokens to generate.
     * @return The generated tokens, not including the prompt.
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
            generated.push_back(
                static_cast<std::uint32_t>(inference_engine::argmax(logits)));
            if (generated.size() == new_tokens) {
                return generated;
            }
            logits = forward({generated.back()});
        }
    }

    /** Clear every rank's cache and return to position zero. */
    void reset() {
        const inference_engine::GptNeoConfig& config = weights_.config;
        const std::size_t head_dim = config.hidden_size / config.num_heads;
        const Tensor<Scalar> empty{{0, head_dim}, {}};
        caches_.assign(
            config.num_layers,
            std::vector<HeadCache>(config.num_heads, HeadCache{empty, empty}));
        position_ = 0;
        syncs_.clear();
    }

    /// @return The syncs of the last forward() call, in order.
    [[nodiscard]] const std::vector<Sync>& syncs() const { return syncs_; }

  private:
    /// One head's cached keys and values, [cached_len, head_dim] each.
    struct HeadCache {
        Tensor<Scalar> key;   ///< Cached keys.
        Tensor<Scalar> value; ///< Cached values.
    };

    /// A projection's weights: float, an optional W8A8 copy, optional bias.
    struct Projection {
        const Tensor<Scalar>& weight;             ///< [out, in].
        const QuantizedWeight<Scalar>* quantized; ///< W8A8 copy, or null.
        const Tensor<Scalar>* bias;               ///< [out], or null.
    };

    [[nodiscard]] Scalar eps() const {
        return static_cast<Scalar>(weights_.config.layer_norm_eps);
    }

    void record(Sync::Kind kind, std::size_t rows, std::size_t width) {
        syncs_.push_back({kind, rows, width});
    }

    /// Outputs @p outputs of @p projection from the whole @p input, with
    /// their bias: one rank's share of a column split.
    static Tensor<Scalar> column_shard(const Tensor<Scalar>& input,
                                       const Projection& projection,
                                       Range outputs) {
        std::optional<Tensor<Scalar>> bias;
        if (projection.bias != nullptr) {
            bias = slice_vector(*projection.bias, outputs);
        }
        const OptionalBias<Scalar> bias_ref =
            bias ? OptionalBias<Scalar>(std::cref(*bias)) : std::nullopt;
        if (projection.quantized != nullptr) {
            return inference_engine::linear<Scalar>(
                input, slice_rows(*projection.quantized, outputs), bias_ref);
        }
        return inference_engine::linear<Scalar>(
            input, slice_rows(projection.weight, outputs), bias_ref);
    }

    /// Partial output of @p projection from input columns @p inputs, with
    /// no bias: one rank's share of a row split.
    static Tensor<Scalar> row_shard(const Tensor<Scalar>& input_slice,
                                    const Projection& projection,
                                    Range inputs) {
        if (projection.quantized != nullptr) {
            return inference_engine::linear<Scalar>(
                input_slice, slice_columns(*projection.quantized, inputs));
        }
        return inference_engine::linear<Scalar>(
            input_slice, slice_columns(projection.weight, inputs));
    }

    /** Combine the ranks' slices through a projection that reads all of
     * them: the attention output projection or c_proj.
     *
     * @param slices Rank r's slice of the projection's input.
     * @param inputs Rank r's range of the projection's input columns.
     * @param projection The projection.
     * @return The projection's whole output, which every rank then holds.
     */
    Tensor<Scalar> combine(const std::vector<Tensor<Scalar>>& slices,
                           const std::vector<Range>& inputs,
                           const Projection& projection) {
        const std::size_t rows = slices.front().shape.at(0);
        const std::size_t out_features = weight_shape(projection.weight).at(0);
        if (scheme_ == TpScheme::kMegatron) {
            std::vector<Tensor<Scalar>> partials;
            for (std::size_t rank = 0; rank < ranks_; ++rank) {
                partials.push_back(
                    row_shard(slices[rank], projection, inputs[rank]));
            }
            record(Sync::Kind::kSum, rows, out_features);
            return sum_in_rank_order(
                partials,
                projection.bias != nullptr
                    ? OptionalBias<Scalar>(std::cref(*projection.bias))
                    : std::nullopt);
        }
        const Tensor<Scalar> whole = concat_columns(slices);
        record(Sync::Kind::kGather, rows, whole.shape[1]);
        std::vector<Tensor<Scalar>> outputs;
        for (std::size_t rank = 0; rank < ranks_; ++rank) {
            outputs.push_back(column_shard(
                whole, projection, shard_range(out_features, ranks_, rank)));
        }
        record(Sync::Kind::kGather, rows, out_features);
        return concat_columns(outputs);
    }

    Tensor<Scalar> layer_forward(std::size_t layer, const Tensor<Scalar>& x) {
        const inference_engine::GptNeoConfig& config = weights_.config;
        const inference_engine::LayerWeights<Scalar>& weights =
            weights_.layers[layer];
        const inference_engine::AttentionWeights<Scalar>& attention =
            weights.attention;
        const inference_engine::MlpWeights<Scalar>& mlp = weights.mlp;
        const auto quantized =
            [](const auto& copies,
               std::size_t index) -> const QuantizedWeight<Scalar>* {
            return copies.has_value() ? &copies->at(index) : nullptr;
        };
        const auto bias =
            [](const Tensor<Scalar>& values) -> const Tensor<Scalar>* {
            return values.values.empty() ? nullptr : &values;
        };

        // Attention: each rank projects, caches and attends its own heads.
        const Tensor<Scalar> normed = inference_engine::layernorm<Scalar>(
            x, weights.ln_1_weight, std::cref(weights.ln_1_bias), eps());
        const std::size_t rows = x.shape[0];
        const std::size_t head_dim = config.hidden_size / config.num_heads;
        const std::size_t key_len = position_ + rows;
        const Tensor<bool> visible = inference_engine::causal_visibility(
            rows, key_len, position_,
            config.is_local(layer) ? std::optional(config.window_size)
                                   : std::nullopt);
        std::vector<Tensor<Scalar>> contexts;
        std::vector<Range> context_columns;
        for (std::size_t rank = 0; rank < ranks_; ++rank) {
            const Range heads = shard_range(config.num_heads, ranks_, rank);
            const Range columns{heads.start * head_dim, heads.count * head_dim};
            const Tensor<Scalar> query =
                column_shard(normed,
                             {attention.query_weights,
                              quantized(attention.quantized, 0), nullptr},
                             columns);
            const Tensor<Scalar> key =
                column_shard(normed,
                             {attention.key_weights,
                              quantized(attention.quantized, 1), nullptr},
                             columns);
            const Tensor<Scalar> value =
                column_shard(normed,
                             {attention.value_weights,
                              quantized(attention.quantized, 2), nullptr},
                             columns);
            std::vector<Tensor<Scalar>> head_contexts;
            for (std::size_t local = 0; local < heads.count; ++local) {
                const Range head{local * head_dim, head_dim};
                HeadCache& cache = caches_[layer][heads.start + local];
                append_rows(cache.key, slice_columns(key, head));
                append_rows(cache.value, slice_columns(value, head));
                head_contexts.push_back(inference_engine::attend_head(
                    slice_columns(query, head), cache.key, cache.value,
                    visible));
            }
            contexts.push_back(concat_columns(head_contexts));
            context_columns.push_back(columns);
        }
        const Tensor<Scalar> attended = inference_engine::residual_add(
            x, combine(contexts, context_columns,
                       {attention.output_weights,
                        quantized(attention.quantized, 3),
                        bias(attention.output_bias)}));

        // MLP: each rank computes its columns of c_fc and their GELU.
        const Tensor<Scalar> normed_2 = inference_engine::layernorm<Scalar>(
            attended, weights.ln_2_weight, std::cref(weights.ln_2_bias), eps());
        std::vector<Tensor<Scalar>> hidden;
        std::vector<Range> hidden_columns;
        for (std::size_t rank = 0; rank < ranks_; ++rank) {
            const Range columns = shard_range(config.inner_size, ranks_, rank);
            hidden.push_back(inference_engine::gelu(
                column_shard(normed_2,
                             {mlp.c_fc_weight, quantized(mlp.quantized, 0),
                              bias(mlp.c_fc_bias)},
                             columns)));
            hidden_columns.push_back(columns);
        }
        return inference_engine::residual_add(
            attended, combine(hidden, hidden_columns,
                              {mlp.c_proj_weight, quantized(mlp.quantized, 1),
                               bias(mlp.c_proj_bias)}));
    }

    /// Append @p fresh's rows after @p stored's.
    static void append_rows(Tensor<Scalar>& stored,
                            const Tensor<Scalar>& fresh) {
        stored.values.insert(stored.values.end(), fresh.values.begin(),
                             fresh.values.end());
        stored.shape[0] += fresh.shape[0];
    }

    const ModelWeights<Scalar>& weights_;
    std::size_t ranks_;
    TpScheme scheme_;
    /// [layer][head]; rank r holds the heads in shard_range(heads, ranks, r).
    std::vector<std::vector<HeadCache>> caches_;
    std::size_t position_ = 0;
    std::vector<Sync> syncs_;
};

} // namespace dist_inference
