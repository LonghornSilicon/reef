#pragma once

#include <cstddef>
#include <optional>

#include "inference_engine/tensor.hpp"

namespace inference_engine {

// TODO: Define projection weights, Q/K/V tensor shapes, and cache storage
// after the shared Tensor representation is agreed.
struct AttentionWeights;
struct Qkv;
struct KvCache;

// @ Attention team
// TODO: Compute separate query, key, and value projections, then reshape by
// head. Match the workloads model's head count and weight layout.
Qkv project_qkv(const Tensor& input, const AttentionWeights& weights);

// @ Attention team
// TODO: Append new keys and values without losing prior decode positions.
// Define cache capacity and behavior when it is exceeded.
void append_kv_cache(KvCache& cache, const Tensor& key, const Tensor& value);

// @ Attention team
// TODO: Mask future positions and, for local attention layers, positions
// outside the sliding window. A missing window means global attention.
// Account for cached-token offsets.
Tensor apply_causal_mask(const Tensor& scores, std::size_t query_start,
                         std::optional<std::size_t> sliding_window);

// @ Attention team
// TODO: Compute attention scores, apply masking and softmax, then combine
// values. Match the GPT-Neo reference's unscaled scores.
Tensor attention_context(const Qkv& qkv, const KvCache& cache,
                         std::optional<std::size_t> sliding_window);

// @ Attention team
// TODO: Compose projections, cache update, attention_context, and output
// projection for one prefill or decode call.
Tensor attention_forward(const Tensor& input, const AttentionWeights& weights,
                         KvCache& cache);

}  // namespace inference_engine
