#pragma once

/** @file
 *  @brief Attention interface and position-mask variants.
 */

#include "../tensor.hpp"
#include "activation.hpp"
#include "matrix.hpp"

#include <algorithm>
#include <cstddef>

namespace inference_engine {

/** Query, key, value, and output projection parameters.
 *
 * @tparam Scalar Tensor value type.
 */
// TODO: Define the representation after the shared Tensor contract is agreed.
template <typename Scalar> struct AttentionWeights;
/** Projected query, key, and value tensors.
 *
 * @tparam Scalar Tensor value type.
 */
template <typename Scalar> struct Qkv;
/** Mutable key and value tensors retained across decode steps.
 *
 * @tparam Scalar Tensor value type.
 */
template <typename Scalar> struct KvCache;

/** Shared attention pipeline with a required position-mask policy.
 *
 * This is abstract because every concrete attention variant must supply a
 * position mask. Weights and decode state remain externally owned.
 *
 * @tparam Scalar Tensor value type.
 */
// @ Attention team
// TODO: Keep weights and decode cache as shared state. Tests can call forward
// repeatedly for prefill and decode instead of passing that state each time.
template <typename Scalar> class Attention {
  public:
    /** Bind externally owned parameters and decode cache.
     *
     * @param weights Immutable projection parameters.
     * @param cache Mutable key and value cache.
     */
    Attention(const AttentionWeights<Scalar>& weights, KvCache<Scalar>& cache);
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

        const std::size_t num_queries = scores.shape[0];
        const std::size_t num_keys = scores.shape[1];
        constexpr Scalar kNegInf = -std::numeric_limits<Scalar>::infinity();

        for (std::size_t i = 0; i < num_queries; ++i) {
            const std::size_t visible_end =
                std::min(query_start + i + 1, num_keys);
            Scalar* row = out.values.data() + (i * num_keys);
            std::fill(row + visible_end, row + num_keys, kNegInf);
        }
        return out;
    }

  private:
    // TODO: Compute and reshape separate query, key, and value projections.
    [[nodiscard]] Qkv<Scalar> project_qkv(const Tensor<Scalar>& input) const;

    // TODO: Append new keys and values; define cache capacity and overflow.
    void append_kv_cache(const Tensor<Scalar>& key,
                         const Tensor<Scalar>& value);

    [[nodiscard]] Tensor<Scalar> context(const Qkv<Scalar>& qkv) const;

    const AttentionWeights<Scalar>& weights_;
    KvCache<Scalar>& cache_;
};

/** Apply a causal mask that permits every earlier key position.
 *
 * @tparam Scalar Tensor value type.
 */
// @ Attention team
template <typename Scalar> class GlobalAttention : public Attention<Scalar> {
  public:
    /** Bind parameters and cache for unrestricted causal attention.
     *
     * @param weights Immutable projection parameters.
     * @param cache Mutable key and value cache.
     */
    GlobalAttention(const AttentionWeights<Scalar>& weights,
                    KvCache<Scalar>& cache);

  protected:
    /** Mask future key positions.
     *
     * @param scores Query-by-key attention scores.
     * @param query_start Position of the first query in the decode stream.
     * @return Causally masked scores.
     */
    [[nodiscard]] Tensor<Scalar>
    mask_scores(const Tensor<Scalar>& scores,
                std::size_t query_start) const override;
};

/** Apply a causal mask limited to a sliding context window.
 *
 * @tparam Scalar Tensor value type.
 */
// @ Attention team
// TODO: Specialize only the attention mask for local GPT-Neo layers.
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
        : window_size_(window_size), cache_(cache), weights_(weights) {}

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

        const std::size_t num_queries = scores.shape[0];
        const std::size_t num_keys = scores.shape[1];
        constexpr Scalar kNegInf = -std::numeric_limits<Scalar>::infinity();

        // context window trim
        for (std::size_t i = 0; i < num_queries; ++i) {
            const std::size_t pos = query_start + i;
            const std::size_t lo =
                (pos + 1 > window_size_) ? pos + 1 - window_size_ : 0;
            Scalar* row = out.values.data() + (i * num_keys);
            std::fill(row, row + std::min(lo, num_keys), kNegInf);
        }
        return out;
    }

  private:
    std::size_t window_size_;
    const AttentionWeights<Scalar>& weights_;
    KvCache<Scalar>& cache_;
};

// Definitions stay header-visible so any scalar type can instantiate them
// once the weight and cache struct representations are defined.

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