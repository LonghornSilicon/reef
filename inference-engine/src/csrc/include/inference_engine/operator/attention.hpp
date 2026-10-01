#pragma once

#include "inference_engine/tensor.hpp"
#include <cstddef>

namespace inference_engine {

// TODO: Define projection weights, Q/K/V tensor shapes, and cache storage
// after the shared Tensor representation is agreed.
struct AttentionWeights;
struct Qkv;
struct KvCache;

// @ Attention team
// TODO: Keep weights and decode cache as shared state. Tests can call forward
// repeatedly for prefill and decode instead of passing that state each time.
class Attention {
  public:
    Attention(const AttentionWeights& weights, KvCache& cache);
    virtual ~Attention() = default;

    // TODO: Compose projection, cache update, masked attention, and output
    // projection for one prefill or decode call.
    Tensor forward(const Tensor& input);

  protected:
    // TODO: Mask future positions using the cached-token offset. Override in
    // LocalAttention to mask positions outside its sliding window too.
    [[nodiscard]] virtual Tensor mask_scores(const Tensor& scores,
                                             std::size_t query_start) const;

  private:
    // TODO: Compute and reshape separate query, key, and value projections.
    [[nodiscard]] Qkv project_qkv(const Tensor& input) const;

    // TODO: Append new keys and values; define cache capacity and overflow.
    void append_kv_cache(const Tensor& key, const Tensor& value);

    // TODO: Use unscaled GPT-Neo scores, masking, softmax, and weighted values.
    [[nodiscard]] Tensor context(const Qkv& qkv) const;

    const AttentionWeights& weights_;
    KvCache& cache_;
};

// @ Attention team
// TODO: Specialize only the attention mask for local GPT-Neo layers.
class LocalAttention : public Attention {
  public:
    LocalAttention(const AttentionWeights& weights, KvCache& cache,
                   std::size_t window_size);

  protected:
    [[nodiscard]] Tensor mask_scores(const Tensor& scores,
                                     std::size_t query_start) const override;

  private:
    std::size_t window_size_;
};

} // namespace inference_engine
