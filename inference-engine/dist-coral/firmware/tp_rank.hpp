#pragma once

// One rank of GPT-Neo split across chips with Megatron tensor parallelism:
// the Coral kernels stitched into what dist_inference::TpGptNeo computes for
// this rank, one token per step.
//
// The host embeds each token and combines the ranks at every sync
// (tp_host_if.hpp). Per layer, this rank normalizes the whole residual
// stream, attends its own heads, and projects their context through its
// input rows of the attention output: a partial for the host to sum. Then it
// adds the host's sum to its residual stream, computes its columns of c_fc
// and their GELU, and projects them through its input rows of c_proj: the
// second partial. dist-coral/sim/multi_sim.cpp checks the logits of every
// step against TpGptNeo bit for bit.

#include "dist_coral/tp_blob.hpp"
#include "dist_coral/tp_host_if.hpp"

#include <cstddef>
#include <cstdint>

namespace firmware {

// Bytes of arena a rank needs to run max_tokens positions.
std::size_t rank_arena_bytes(const dist_coral::RankHeader& header,
                             std::size_t max_tokens);

class TpRank {
  public:
    // Waits for the host to combine the ranks' outputs: called with the
    // output in exchange() and what it is; returns with the host's reply in
    // exchange().
    using Sync = void (*)(dist_coral::Phase phase);

    // Binds a rank blob and carves the KV cache, activations and exchange
    // buffer out of the arena. Check ok() before calling step().
    TpRank(const std::uint8_t* blob, std::size_t max_tokens,
           std::uint8_t* arena, std::size_t arena_size);

    [[nodiscard]] bool ok() const {
        return error_ == reef_kernels::Status::kIdle;
    }
    [[nodiscard]] reef_kernels::Status error() const { return error_; }

    // The buffer the host reads and writes at each sync,
    // [max(hidden, local_vocab)] floats.
    [[nodiscard]] float* exchange() const { return exchange_; }

    // Runs one token at the next position, its embedding already in
    // exchange(). Syncs twice per layer with a partial. With want_logits,
    // leaves this rank's logits [local_vocab] in exchange() at the end.
    void step(bool want_logits, Sync sync);

  private:
    template <typename T> const T* at(std::uint32_t offset) const {
        return reinterpret_cast<const T*>(blob_ + offset);
    }
    void project(const reef_kernels::BlobLinear& weight, std::size_t in,
                 std::size_t out, const float* x, float* y);
    void project_bias(const reef_kernels::BlobLinear& weight,
                      std::uint32_t bias, std::size_t in, std::size_t out,
                      const float* x, float* y);

    const std::uint8_t* blob_;
    const dist_coral::RankHeader* header_;
    const dist_coral::RankLayer* layers_;
    std::size_t max_tokens_;
    std::size_t position_ = 0;
    reef_kernels::Status error_ = reef_kernels::Status::kIdle;

    // KV caches of this rank's heads
    // [layers][local_heads][max_tokens][head_dim].
    float* key_cache_ = nullptr;
    float* value_cache_ = nullptr;
    // One row of activations each.
    float* x_ = nullptr;         // residual stream [hidden]
    float* normed_ = nullptr;    // LayerNorm output [hidden]
    float* query_ = nullptr;     // this rank's heads [local_width]
    float* key_ = nullptr;       // [local_width]
    float* value_ = nullptr;     // [local_width]
    float* context_ = nullptr;   // this rank's heads, merged [local_width]
    float* inner_ = nullptr;     // this rank's MLP columns [local_inner]
    float* scores_ = nullptr;    // [max_tokens]
    float* exchange_ = nullptr;  // [max(hidden, local_vocab)]
    std::int8_t* x_q_ = nullptr; // W8A8 row scratch [max(hidden, local_inner)]
};

// Serves the host (tp_host_if.hpp) until it sends kStop.
void serve(dist_coral::TpMailbox& mailbox);

} // namespace firmware
