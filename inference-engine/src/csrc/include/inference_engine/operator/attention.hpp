#pragma once

/** @file
 *  @brief Attention interface and position-mask variants.
 */

#include "inference_engine/tensor.hpp"

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
    // TODO: Compose projection, cache update, masked attention, and output
    // projection for one prefill or decode call.
    Tensor<Scalar> forward(const Tensor<Scalar>& input);

  protected:
    /** Mask scores according to the concrete attention policy.
     *
     * @param scores Query-by-key attention scores.
     * @param query_start Position of the first query in the decode stream.
     * @return Scores with disallowed positions masked.
     */
    [[nodiscard]] virtual Tensor<Scalar>
    mask_scores(const Tensor<Scalar>& scores,
                std::size_t query_start) const = 0;

  private:
    // TODO: Compute and reshape separate query, key, and value projections.
    [[nodiscard]] Qkv<Scalar> project_qkv(const Tensor<Scalar>& input) const;

    // TODO: Append new keys and values; define cache capacity and overflow.
    void append_kv_cache(const Tensor<Scalar>& key,
                         const Tensor<Scalar>& value);

    // TODO: Define score scaling, then mask, softmax, and weight values.
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
                   KvCache<Scalar>& cache, std::size_t window_size);

  protected:
    /** Mask future keys and keys outside the context window.
     *
     * @param scores Query-by-key attention scores.
     * @param query_start Position of the first query in the decode stream.
     * @return Causally and locally masked scores.
     */
    [[nodiscard]] Tensor<Scalar>
    mask_scores(const Tensor<Scalar>& scores,
                std::size_t query_start) const override;

  private:
    std::size_t window_size_;
};

} // namespace inference_engine
