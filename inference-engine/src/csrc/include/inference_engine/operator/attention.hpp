#pragma once

/** @file
 *  @brief Attention interface and position-mask variants.
 */

#include "inference_engine/operator/activation.hpp"
#include "inference_engine/operator/linear.hpp"
#include "inference_engine/tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
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
    std::size_t num_heads;         ///< Heads that split the model width.
    Tensor<Scalar> query_weights;  ///< [d_model, d_model], no bias.
    Tensor<Scalar> key_weights;    ///< [d_model, d_model], no bias.
    Tensor<Scalar> value_weights;  ///< [d_model, d_model], no bias.
    Tensor<Scalar> output_weights; ///< [d_model, d_model].
    Tensor<Scalar> output_bias;    ///< [d_model].
};

/** Projected query, key, and value tensors for one batch.
 *
 * @tparam Scalar Tensor value type.
 */
template <typename Scalar> struct Qkv {
    Tensor<Scalar> query; ///< [1, heads, query_len, head_dim].
    Tensor<Scalar> key;   ///< [1, heads, query_len, head_dim].
    Tensor<Scalar> value; ///< [1, heads, query_len, head_dim].
};

/** Mutable key and value tensors retained across decode steps.
 *
 * The tensors hold only filled positions, so `key.shape[2]` is the number of
 * cached tokens. An empty cache has shape `[1, heads, 0, head_dim]`, and
 * a default-constructed cache bootstraps its shape from the first append. A
 * disengaged capacity grows on demand.
 *
 * @tparam Scalar Tensor value type.
 */
template <typename Scalar> struct KvCache {
    std::optional<std::size_t> capacity; ///< Optional maximum cached positions.
    Tensor<Scalar> key;                  ///< [1, heads, cached_len, head_dim].
    Tensor<Scalar> value;                ///< [1, heads, cached_len, head_dim].
};

/** Shared attention pipeline with an overridable position-mask policy.
 *
 * The base mask is the unrestricted causal policy; variants override
 * visibility to restrict it further. Weights and decode state remain
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
     * @param input Input rows [query_len, d_model] for prefill or decode.
     * @return Output rows [query_len, d_model].
     */
    Tensor<Scalar> forward(const Tensor<Scalar>& input);

  protected:
    /** Choose the visible key positions for one forward call.
     *
     * The default policy is unrestricted causal attention: each query sees
     * every key up to and including its own position.
     *
     * @param num_queries Query rows in this call.
     * @param num_keys Cached key positions, including this call's.
     * @param query_start Position of the first query in the decode stream.
     * @return [num_queries, num_keys] mask, true where the key is visible.
     */
    [[nodiscard]] virtual Tensor<bool>
    visibility(std::size_t num_queries, std::size_t num_keys,
               std::size_t query_start) const {
        Tensor<bool> visible{{num_queries, num_keys},
                             std::vector<bool>(num_queries * num_keys, false)};
        for (std::size_t row = 0; row < num_queries; ++row) {
            const std::size_t visible_end =
                std::min(query_start + row + 1, num_keys);
            for (std::size_t key = 0; key < visible_end; ++key) {
                visible.values.at((row * num_keys) + key) = true;
            }
        }
        return visible;
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

/** Attend causally with the base policy: every earlier key is visible.
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
     * @throws std::invalid_argument If @p window_size is zero, which would
     * leave queries with no visible position.
     */
    LocalAttention(const AttentionWeights<Scalar>& weights,
                   KvCache<Scalar>& cache, std::size_t window_size)
        : Attention<Scalar>(weights, cache), window_size_(window_size) {
        if (window_size == 0) {
            throw std::invalid_argument(
                "LocalAttention window must hold at least one position");
        }
    }

  protected:
    /** Restrict causal visibility to the sliding context window.
     *
     * The window includes the query's own position.
     *
     * @param num_queries Query rows in this call.
     * @param num_keys Cached key positions, including this call's.
     * @param query_start Position of the first query in the decode stream.
     * @return [num_queries, num_keys] mask, true where the key is visible.
     */
    [[nodiscard]] Tensor<bool>
    visibility(std::size_t num_queries, std::size_t num_keys,
               std::size_t query_start) const override {
        Tensor<bool> visible =
            Attention<Scalar>::visibility(num_queries, num_keys, query_start);
        for (std::size_t row = 0; row < num_queries; ++row) {
            const std::size_t pos = query_start + row;
            const std::size_t window_start =
                (pos + 1 > window_size_) ? pos + 1 - window_size_ : 0;
            for (std::size_t key = 0; key < std::min(window_start, num_keys);
                 ++key) {
                visible.values.at((row * num_keys) + key) = false;
            }
        }
        return visible;
    }

  private:
    std::size_t window_size_;
};

// Definitions stay header-visible so any scalar type can instantiate them
// once the weight and cache struct representations are defined.
template <typename Scalar>
Qkv<Scalar> Attention<Scalar>::project_qkv(const Tensor<Scalar>& input) const {
    if (!valid_matrix(input) || weights_.num_heads == 0 ||
        input.shape[1] % weights_.num_heads != 0) {
        throw std::invalid_argument(
            "project_qkv requires [length, d_model] input and valid heads");
    }
    const std::size_t length = input.shape[0];
    const std::size_t hidden = input.shape[1];
    const std::size_t head_dim = hidden / weights_.num_heads;

    // Project at full width, then split columns into heads:
    // [length, d_model] -> [1, heads, length, head_dim].
    const auto split_heads = [&](const Tensor<Scalar>& weight) {
        const Tensor<Scalar> projected = linear<Scalar>(input, weight);
        return transpose(
            reshape(projected, {1, length, weights_.num_heads, head_dim}), 1,
            2);
    };
    return {split_heads(weights_.query_weights),
            split_heads(weights_.key_weights),
            split_heads(weights_.value_weights)};
}

template <typename Scalar>
void Attention<Scalar>::append_kv_cache(const Tensor<Scalar>& key,
                                        const Tensor<Scalar>& value) {
    if (key.shape.size() != 4 || key.shape[0] != 1 ||
        value.shape != key.shape || !valid_tensor(key) ||
        !valid_tensor(value)) {
        throw std::invalid_argument("KV cache append shape mismatch");
    }
    // A default-constructed cache bootstraps its shape from the first append.
    if (cache_.key.shape.empty() && cache_.value.shape.empty() &&
        cache_.key.values.empty() && cache_.value.values.empty()) {
        const std::vector<std::size_t> empty_shape{key.shape[0], key.shape[1],
                                                   0, key.shape[3]};
        cache_.key = Tensor<Scalar>{empty_shape, {}};
        cache_.value = Tensor<Scalar>{empty_shape, {}};
    }
    if (cache_.key.shape.size() != 4 ||
        cache_.value.shape != cache_.key.shape) {
        throw std::invalid_argument("KV cache append shape mismatch");
    }
    const std::vector<std::size_t>& cached = cache_.key.shape;
    if (key.shape[0] != cached[0] || key.shape[1] != cached[1] ||
        key.shape[3] != cached[3] ||
        (cached[2] == 0 &&
         (!cache_.key.values.empty() || !cache_.value.values.empty())) ||
        (cached[2] != 0 &&
         (!valid_tensor(cache_.key) || !valid_tensor(cache_.value)))) {
        throw std::invalid_argument("KV cache append shape mismatch");
    }
    const std::size_t old_len = cached[2];
    const std::size_t new_len = key.shape[2];
    if (new_len > std::numeric_limits<std::size_t>::max() - old_len) {
        throw std::length_error("KV cache length overflow");
    }
    const std::size_t total_len = old_len + new_len;
    if (cache_.capacity && total_len > *cache_.capacity) {
        throw std::length_error("KV cache capacity exceeded");
    }

    // Each (batch, head) block is contiguous, so new positions go at the end
    // of every block rather than at the end of the whole tensor.
    const std::size_t blocks = key.shape[0] * key.shape[1];
    const std::size_t head_dim = key.shape[3];
    const auto extend = [&](Tensor<Scalar>& stored,
                            const Tensor<Scalar>& fresh) {
        const std::size_t old_block = old_len * head_dim;
        const std::size_t new_block = new_len * head_dim;
        std::vector<Scalar> merged;
        merged.reserve(stored.values.size() + fresh.values.size());
        for (std::size_t block = 0; block < blocks; ++block) {
            for (std::size_t index = 0; index < old_block; ++index) {
                merged.push_back(stored.values.at((block * old_block) + index));
            }
            for (std::size_t index = 0; index < new_block; ++index) {
                merged.push_back(fresh.values.at((block * new_block) + index));
            }
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

    // Merge heads and project to 2-D output rows.
    Tensor<Scalar> swapped = transpose(mixed, 1, 2);
    const std::size_t batch = swapped.shape[0];
    assert(batch == 1);
    const std::size_t query_len = swapped.shape[1];
    const std::size_t hidden = swapped.shape[2] * swapped.shape[3];
    const Tensor<Scalar> merged = reshape(swapped, {query_len, hidden});
    return linear<Scalar>(merged, weights_.output_weights,
                          std::cref(weights_.output_bias));
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
    assert(batch == 1 && key_len >= query_len);

    const auto head_matrix = [](const Tensor<Scalar>& tensor,
                                std::size_t batch_index, std::size_t head) {
        const std::size_t rows = tensor.shape.at(2);
        const std::size_t columns = tensor.shape.at(3);
        const std::size_t start =
            ((batch_index * tensor.shape.at(1)) + head) * rows * columns;
        std::vector<Scalar> values;
        values.reserve(rows * columns);
        for (std::size_t index = 0; index < rows * columns; ++index) {
            values.push_back(tensor.values.at(start + index));
        }
        return Tensor<Scalar>{{rows, columns}, std::move(values)};
    };
    const auto store_head_matrix = [](Tensor<Scalar>& destination,
                                      const Tensor<Scalar>& matrix,
                                      std::size_t batch_index,
                                      std::size_t head) {
        const std::size_t rows = destination.shape.at(2);
        const std::size_t columns = destination.shape.at(3);
        assert(matrix.shape == (std::vector<std::size_t>{rows, columns}));
        const std::size_t start =
            ((batch_index * destination.shape.at(1)) + head) * rows * columns;
        for (std::size_t index = 0; index < rows * columns; ++index) {
            destination.values.at(start + index) = matrix.values.at(index);
        }
    };

    // Raw scores matmul
    Tensor<Scalar> scores{
        {batch, heads, query_len, key_len},
        std::vector<Scalar>(batch * heads * query_len * key_len, Scalar{0}),
    };

    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < heads; ++h) {
            const Tensor<Scalar> query_slice = head_matrix(query, b, h);
            const Tensor<Scalar> key_slice = head_matrix(key, b, h);
            const Tensor<Scalar> head_scores =
                matmul(query_slice, transpose(key_slice, 0, 1));
            store_head_matrix(scores, head_scores, b, h);
        }
    }

    // Mask and softmax
    const std::size_t query_start = key_len - query_len;
    const Tensor<Scalar> probs =
        masked_softmax(scores, visibility(query_len, key_len, query_start));

    // Probabilities weighted sum matmul
    Tensor<Scalar> mixed{
        {batch, heads, query_len, head_dim},
        std::vector<Scalar>(batch * heads * query_len * head_dim, Scalar{0}),
    };

    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < heads; ++h) {
            const Tensor<Scalar> probs_slice = head_matrix(probs, b, h);
            const Tensor<Scalar> value_slice = head_matrix(value, b, h);
            const Tensor<Scalar> head_mix = matmul(probs_slice, value_slice);
            store_head_matrix(mixed, head_mix, b, h);
        }
    }

    // Return mixed
    return mixed;
}

} // namespace inference_engine
