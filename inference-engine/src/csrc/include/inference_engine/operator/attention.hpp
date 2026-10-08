#pragma once

/** @file
 *  @brief Attention interface and position-mask variants.
 */

#include "inference_engine/operator/activation.hpp"
#include "inference_engine/operator/linear.hpp"
#include "inference_engine/tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace inference_engine {

/** Query, key, value, and output projection parameters.
 *
 * Every head shares one full-width projection per role; heads are split
 * after projection. Weights use the Hugging Face `[out, in]` layout, so each
 * projection computes `input @ weight.T`.
 *
 * @tparam Scalar Tensor value type.
 */
template <typename Scalar> struct AttentionWeights {
    std::size_t num_heads = 0;     ///< Heads that split the model width.
    Tensor<Scalar> query_weights;  ///< [d_model, d_model], no bias.
    Tensor<Scalar> key_weights;    ///< [d_model, d_model], no bias.
    Tensor<Scalar> value_weights;  ///< [d_model, d_model], no bias.
    Tensor<Scalar> output_weights; ///< [d_model, d_model].
    Tensor<Scalar> output_bias;    ///< [d_model].
};

/** Projected query, key, and value tensors.
 *
 * @tparam Scalar Tensor value type.
 */
template <typename Scalar> struct Qkv {
    Tensor<Scalar> query; ///< [batch, heads, query_len, head_dim].
    Tensor<Scalar> key;   ///< [batch, heads, query_len, head_dim].
    Tensor<Scalar> value; ///< [batch, heads, query_len, head_dim].
};

/** Mutable key and value tensors retained across decode steps.
 *
 * The tensors hold only filled positions, so `key.shape[2]` is the number of
 * cached tokens. An empty cache has shape `[batch, heads, 0, head_dim]`, and
 * a default-constructed cache bootstraps its shape from the first append. A
 * capacity of 0 grows on demand.
 *
 * @tparam Scalar Tensor value type.
 */
template <typename Scalar> struct KvCache {
    std::size_t capacity = 0; ///< Maximum cached positions per head.
    Tensor<Scalar> key;       ///< [batch, heads, cached_len, head_dim].
    Tensor<Scalar> value;     ///< [batch, heads, cached_len, head_dim].
};

/** Shared attention pipeline with an overridable position-mask policy.
 *
 * The base mask is the unrestricted causal policy; variants override
 * mask_scores to restrict it further. Weights and decode state remain
 * externally owned.
 *
 * @tparam Scalar Tensor value type.
 */
template <typename Scalar> class Attention {
  public:
    /** Bind externally owned parameters and decode cache.
     *
     * @param weights Immutable projection parameters.
     * @param cache Mutable key and value cache.
     */
    Attention(const AttentionWeights<Scalar>& weights, KvCache<Scalar>& cache)
        : weights_(weights), cache_(cache) {}
    /// Destroy through the abstract base class.
    virtual ~Attention() = default;

    /** Process input positions using the variant's mask policy.
     *
     * @param input Input tensor for prefill or decode.
     * @return Attention output tensor.
     */
    Tensor<Scalar> forward(const Tensor<Scalar>& input);

  protected:
    /** Mask scores according to the concrete attention policy.
     *
     * @param scores Query-by-key attention scores.
     * @param query_start Position of the first query in the decode stream.
     * @return Scores with disallowed positions masked.
     */
    [[nodiscard]] virtual Tensor<Scalar>
    mask_scores(const Tensor<Scalar>& scores, std::size_t query_start) const {
        Tensor<Scalar> out = scores;

        // Scores are [..., queries, keys]; mask every leading (batch, head)
        // block the same way.
        const std::size_t rank = scores.shape.size();
        const std::size_t num_queries = scores.shape[rank - 2];
        const std::size_t num_keys = scores.shape[rank - 1];
        const std::size_t num_rows = scores.values.size() / num_keys;
        constexpr auto kMasked = masked_score<Scalar>();

        for (std::size_t r = 0; r < num_rows; ++r) {
            const std::size_t i = r % num_queries;
            const std::size_t visible_end =
                std::min(query_start + i + 1, num_keys);
            Scalar* row = out.values.data() + (r * num_keys);
            std::fill(row + visible_end, row + num_keys, kMasked);
        }
        return out;
    }

    /// Externally owned projection parameters.
    const AttentionWeights<Scalar>& weights_;
    /// Externally owned key and value cache, updated on each forward.
    KvCache<Scalar>& cache_;

  private:
    [[nodiscard]] Qkv<Scalar> project_qkv(const Tensor<Scalar>& input) const;

    void append_kv_cache(const Tensor<Scalar>& key,
                         const Tensor<Scalar>& value);

    [[nodiscard]] Tensor<Scalar> context(const Qkv<Scalar>& qkv) const;
};

/** Apply a causal mask that permits every earlier key position.
 *
 * @tparam Scalar Tensor value type.
 */
template <typename Scalar> class GlobalAttention : public Attention<Scalar> {
  public:
    /** Bind parameters and cache for unrestricted causal attention.
     *
     * @param weights Immutable projection parameters.
     * @param cache Mutable key and value cache.
     */
    GlobalAttention(const AttentionWeights<Scalar>& weights,
                    KvCache<Scalar>& cache)
        : Attention<Scalar>(weights, cache) {}
};

/** Apply a causal mask limited to a sliding context window.
 *
 * @tparam Scalar Tensor value type.
 */
template <typename Scalar> class LocalAttention : public Attention<Scalar> {
  public:
    /** Bind parameters, cache, and context-window length.
     *
     * @param weights Immutable projection parameters.
     * @param cache Mutable key and value cache.
     * @param window_size Maximum number of visible key positions.
     */
    LocalAttention(const AttentionWeights<Scalar>& weights,
                   KvCache<Scalar>& cache, std::size_t window_size)
        : Attention<Scalar>(weights, cache), window_size_(window_size) {}

  protected:
    /** Mask future keys and keys outside the context window.
     *
     * @param scores Query-by-key attention scores.
     * @param query_start Position of the first query in the decode stream.
     * @return Causally and locally masked scores.
     */
    [[nodiscard]] Tensor<Scalar>
    mask_scores(const Tensor<Scalar>& scores,
                std::size_t query_start) const override {
        // causal part
        Tensor<Scalar> out =
            Attention<Scalar>::mask_scores(scores, query_start);

        const std::size_t rank = scores.shape.size();
        const std::size_t num_queries = scores.shape[rank - 2];
        const std::size_t num_keys = scores.shape[rank - 1];
        const std::size_t num_rows = scores.values.size() / num_keys;
        constexpr auto kMasked = masked_score<Scalar>();

        // context window trim
        for (std::size_t r = 0; r < num_rows; ++r) {
            const std::size_t pos = query_start + (r % num_queries);
            const std::size_t lo =
                (pos + 1 > window_size_) ? pos + 1 - window_size_ : 0;
            Scalar* row = out.values.data() + (r * num_keys);
            std::fill(row, row + std::min(lo, num_keys), kMasked);
        }
        return out;
    }

  private:
    std::size_t window_size_;
};

// Definitions stay header-visible so any scalar type can instantiate them
// once the weight and cache struct representations are defined.
template <typename Scalar>
Qkv<Scalar> Attention<Scalar>::project_qkv(const Tensor<Scalar>& input) const {
    if (input.shape.size() != 3 || weights_.num_heads == 0 ||
        input.shape[2] % weights_.num_heads != 0) {
        throw std::invalid_argument(
            "project_qkv requires [batch, length, d_model] input");
    }
    const std::size_t batch = input.shape[0];
    const std::size_t length = input.shape[1];
    const std::size_t hidden = input.shape[2];
    const std::size_t head_dim = hidden / weights_.num_heads;
    const Tensor<Scalar> rows = reshape(input, {batch * length, hidden});

    // Project at full width, then split columns into heads:
    // [batch * length, d_model] -> [batch, heads, length, head_dim].
    const auto split_heads = [&](const Tensor<Scalar>& weight) {
        const Tensor<Scalar> projected = linear<Scalar>(rows, weight);
        return transpose(
            reshape(projected, {batch, length, weights_.num_heads, head_dim}),
            1, 2);
    };
    return {split_heads(weights_.query_weights),
            split_heads(weights_.key_weights),
            split_heads(weights_.value_weights)};
}

template <typename Scalar>
void Attention<Scalar>::append_kv_cache(const Tensor<Scalar>& key,
                                        const Tensor<Scalar>& value) {
    if (key.shape.size() != 4 || value.shape != key.shape) {
        throw std::invalid_argument("KV cache append shape mismatch");
    }
    // A default-constructed cache bootstraps its shape from the first append.
    if (cache_.key.shape.size() != 4) {
        const std::vector<std::size_t> empty_shape{key.shape[0], key.shape[1],
                                                   0, key.shape[3]};
        cache_.key = Tensor<Scalar>{empty_shape, {}};
        cache_.value = Tensor<Scalar>{empty_shape, {}};
    }
    const std::vector<std::size_t>& cached = cache_.key.shape;
    if (key.shape[0] != cached[0] || key.shape[1] != cached[1] ||
        key.shape[3] != cached[3]) {
        throw std::invalid_argument("KV cache append shape mismatch");
    }
    const std::size_t old_len = cached[2];
    const std::size_t new_len = key.shape[2];
    const std::size_t total_len = old_len + new_len;
    if (cache_.capacity != 0 && total_len > cache_.capacity) {
        throw std::length_error("KV cache capacity exceeded");
    }

    // Each (batch, head) block is contiguous, so new positions go at the end
    // of every block rather than at the end of the whole tensor.
    const std::size_t blocks = key.shape[0] * key.shape[1];
    const std::size_t head_dim = key.shape[3];
    const auto extend = [&](Tensor<Scalar>& stored,
                            const Tensor<Scalar>& fresh) {
        const auto old_block = static_cast<std::ptrdiff_t>(old_len * head_dim);
        const auto new_block = static_cast<std::ptrdiff_t>(new_len * head_dim);
        std::vector<Scalar> merged;
        merged.reserve(stored.values.size() + fresh.values.size());
        for (std::size_t block = 0; block < blocks; ++block) {
            const auto index = static_cast<std::ptrdiff_t>(block);
            const auto old_begin = stored.values.begin() + (index * old_block);
            const auto new_begin = fresh.values.begin() + (index * new_block);
            merged.insert(merged.end(), old_begin, old_begin + old_block);
            merged.insert(merged.end(), new_begin, new_begin + new_block);
        }
        stored.values = std::move(merged);
        stored.shape[2] = total_len;
    };
    extend(cache_.key, key);
    extend(cache_.value, value);
}

template <typename Scalar>
Tensor<Scalar> Attention<Scalar>::forward(const Tensor<Scalar>& input) {
    // Project input to QKV and append to cache
    Qkv<Scalar> qkv = project_qkv(input);
    append_kv_cache(qkv.key, qkv.value);

    // Compute context
    Tensor<Scalar> mixed = context(qkv);

    // Merge heads and project to output
    Tensor<Scalar> swapped = transpose(mixed, 1, 2);
    const std::size_t batch = swapped.shape[0];
    const std::size_t query_len = swapped.shape[1];
    const std::size_t hidden = swapped.shape[2] * swapped.shape[3];
    Tensor<Scalar> merged = reshape(swapped, {batch * query_len, hidden});
    Tensor<Scalar> out_proj = linear<Scalar>(merged, weights_.output_weights,
                                             std::cref(weights_.output_bias));

    // Reshape output to original shape
    return reshape(out_proj, {batch, query_len, hidden});
}

template <typename Scalar>
Tensor<Scalar> Attention<Scalar>::context(const Qkv<Scalar>& qkv) const {
    const Tensor<Scalar>& query = qkv.query;
    const Tensor<Scalar>& key = cache_.key;
    const Tensor<Scalar>& value = cache_.value;

    const std::size_t batch = query.shape[0];
    const std::size_t heads = query.shape[1];
    const std::size_t query_len = query.shape[2];
    const std::size_t head_dim = query.shape[3];
    const std::size_t key_len = key.shape[2];

    // Raw scores matmul
    Tensor<Scalar> scores{
        {batch, heads, query_len, key_len},
        std::vector<Scalar>(batch * heads * query_len * key_len, Scalar{0}),
    };

    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < heads; ++h) {
            const std::size_t q_off = ((b * heads) + h) * query_len * head_dim;
            const std::size_t k_off = ((b * heads) + h) * key_len * head_dim;
            const Tensor<Scalar> query_slice{
                {query_len, head_dim},
                {
                    query.values.begin() + q_off,
                    query.values.begin() + q_off + (query_len * head_dim),
                },
            };
            const Tensor<Scalar> key_slice{
                {key_len, head_dim},
                {
                    key.values.begin() + k_off,
                    key.values.begin() + k_off + (key_len * head_dim),
                },
            };

            const Tensor<Scalar> head_scores =
                matmul(query_slice, transpose(key_slice, 0, 1));

            std::copy(head_scores.values.begin(), head_scores.values.end(),
                      scores.values.begin() +
                          (((b * heads) + h) * query_len * key_len));
        }
    }

    // Mask and softmax
    const std::size_t query_start = key_len - query_len;
    const Tensor<Scalar> masked = mask_scores(scores, query_start);
    const Tensor<Scalar> probs = softmax(masked, 3);

    // Probabilities weighted sum matmul
    Tensor<Scalar> mixed{
        {batch, heads, query_len, head_dim},
        std::vector<Scalar>(batch * heads * query_len * head_dim, Scalar{0}),
    };

    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < heads; ++h) {
            const std::size_t w_off = ((b * heads) + h) * query_len * key_len;
            const std::size_t v_off = ((b * heads) + h) * key_len * head_dim;
            const Tensor<Scalar> probs_slice{
                {query_len, key_len},
                {
                    probs.values.begin() + w_off,
                    probs.values.begin() + w_off + (query_len * key_len),
                },
            };
            const Tensor<Scalar> value_slice{
                {key_len, head_dim},
                {
                    value.values.begin() + v_off,
                    value.values.begin() + v_off + (key_len * head_dim),
                },
            };

            const Tensor<Scalar> head_mix = matmul(probs_slice, value_slice);

            std::copy(head_mix.values.begin(), head_mix.values.end(),
                      mixed.values.begin() +
                          (((b * heads) + h) * query_len * head_dim));
        }
    }

    // Return mixed
    return mixed;
}

} // namespace inference_engine
