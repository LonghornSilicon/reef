#pragma once

// Byte layout of one rank's share of a GPT-Neo model split across chips with
// Megatron tensor parallelism: what the host loads into each chip's memory.
// dist-coral/host/pack_rank.hpp writes it, cutting the weights exactly as
// dist_inference::TpGptNeo does, and dist-coral/firmware/ reads it. The
// conventions are model_blob.hpp's: little-endian, offsets in bytes from the
// start of the blob, 16-byte aligned, linear weights stored [in][out].
//
// Rank r holds:
//   - heads shard_range(num_heads, ranks, r): their query, key and value
//     columns, and the matching input rows of the attention output;
//   - MLP columns shard_range(inner, ranks, r): those columns of c_fc and
//     its bias, and the matching input rows of c_proj;
//   - vocabulary rows shard_range(vocab, ranks, r) of the tied lm_head;
//   - every LayerNorm, since each rank normalizes the whole residual stream.
// The host keeps the rest: the embeddings (it embeds every token) and the
// attention output and c_proj biases (it adds them once, after summing the
// ranks' partials).

#include "reef_kernels/model_blob.hpp"

#include <cstdint>

namespace dist_coral {

inline constexpr std::uint32_t kRankBlobMagic = 0x50544552; // "RETP"
inline constexpr std::uint32_t kRankBlobVersion = 1;

struct RankLayer {
    std::uint32_t ln_1_weight, ln_1_bias; // float [hidden]
    reef_kernels::BlobLinear query, key;  // [hidden][local_width]
    reef_kernels::BlobLinear value;       // [hidden][local_width]
    reef_kernels::BlobLinear output;      // [local_width][hidden], no bias
    std::uint32_t ln_2_weight, ln_2_bias; // float [hidden]
    reef_kernels::BlobLinear fc;          // [hidden][local_inner]
    std::uint32_t fc_bias;                // float [local_inner]
    reef_kernels::BlobLinear proj;        // [local_inner][hidden], no bias
    std::uint32_t is_local;               // 1: sliding-window attention
};

struct RankHeader {
    std::uint32_t magic;
    std::uint32_t version;
    reef_kernels::Precision precision;
    std::uint32_t rank;
    std::uint32_t ranks;
    std::uint32_t max_positions;
    std::uint32_t hidden_size;
    std::uint32_t num_layers;
    std::uint32_t head_dim;
    std::uint32_t local_heads; // heads on this rank
    std::uint32_t local_inner; // MLP columns on this rank
    std::uint32_t vocab_start; // first lm_head row on this rank
    std::uint32_t local_vocab; // lm_head rows on this rank
    std::uint32_t window_size;
    float layer_norm_eps;
    std::uint32_t ln_f_weight; // float [hidden]
    std::uint32_t ln_f_bias;   // float [hidden]
    std::uint32_t lm_head;     // float [local_vocab][hidden], wte's rows
    std::uint32_t layers;      // RankLayer [num_layers]
    std::uint32_t total_bytes;
};

} // namespace dist_coral
