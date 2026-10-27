#pragma once

// GPT-Neo on the Coral core: the kernels stitched into the model.
//
// This mirrors GPTNeo in include/inference_engine/model.hpp, one
// token per step. The reference may process a whole prompt at once; every
// operator treats rows independently, so feeding the prompt one token at a
// time produces the same bits. coral/tests/test_gpt_neo.cc checks the logits
// of every step and the generated tokens against the reference.

#include "reef_kernels/host_if.hpp"
#include "reef_kernels/model_blob.hpp"

#include <cstddef>
#include <cstdint>

namespace firmware {

// Bytes of arena a model needs to run max_tokens positions.
std::size_t arena_bytes(const reef_kernels::BlobHeader& header,
                        std::size_t max_tokens);

class GptNeo {
  public:
    // Binds a packed model and carves the KV cache and activations out of
    // the arena. Check ok() before calling step().
    GptNeo(const std::uint8_t* blob, std::size_t max_tokens,
           std::uint8_t* arena, std::size_t arena_size);

    [[nodiscard]] bool ok() const {
        return error_ == reef_kernels::Status::kIdle;
    }
    [[nodiscard]] reef_kernels::Status error() const { return error_; }

    // Runs one token at the next position. With want_logits, also applies
    // ln_f and lm_head and returns the argmax, the greedy next token;
    // otherwise skips them (a prompt token before the last) and returns 0.
    std::uint32_t step(std::uint32_t token, bool want_logits);

    // The logits of the last step that asked for them, [vocab_size].
    [[nodiscard]] const float* logits() const { return logits_; }

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
    const reef_kernels::BlobHeader* header_;
    const reef_kernels::BlobLayer* layers_;
    std::size_t max_tokens_;
    std::size_t position_ = 0;
    reef_kernels::Status error_ = reef_kernels::Status::kIdle;

    // KV caches [layers][heads][max_tokens][head_dim].
    float* key_cache_ = nullptr;
    float* value_cache_ = nullptr;
    // One row of activations each.
    float* x_ = nullptr;         // residual stream [hidden]
    float* normed_ = nullptr;    // LayerNorm output [hidden]
    float* query_ = nullptr;     // [hidden]
    float* key_ = nullptr;       // [hidden]
    float* value_ = nullptr;     // [hidden]
    float* context_ = nullptr;   // merged heads [hidden]
    float* sublayer_ = nullptr;  // attention or MLP output [hidden]
    float* inner_ = nullptr;     // MLP inner [inner]
    float* scores_ = nullptr;    // [max_tokens]
    float* logits_ = nullptr;    // [vocab]
    std::int8_t* x_q_ = nullptr; // W8A8 row scratch [max(hidden, inner)]
};

// Serves the generate request in the mailbox (host_if.hpp): feeds the
// prompt, appends new_tokens greedy tokens after it, and sets status.
void serve(reef_kernels::Mailbox& mailbox);

} // namespace firmware
