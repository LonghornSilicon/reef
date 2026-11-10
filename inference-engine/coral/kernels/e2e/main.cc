// The whole model: prefill the mailbox's prompt in blocks of kBlockTokens,
// then greedy decode, one token per step. mcycle is read after every kernel
// (7 per layer, 1 per embedding, 2 per lm_head step).
//
// E2E_PROMPT / E2E_NEW_TOKENS set the mailbox's default request.
// E2E_START_POS=N skips prefill and decodes at position N, as if the KV cache
// already held N positions.

#include "common.hpp"

using namespace npu;

#ifndef E2E_PROMPT
#define E2E_PROMPT 32
#endif
#ifndef E2E_NEW_TOKENS
#define E2E_NEW_TOKENS 8
#endif
// E2E_START_POS: decode at this position with no prefill, as if a prompt of
// that length were already in the KV cache (the timing does not depend on what
// the cache holds). 0: prefill the prompt first.
#ifndef E2E_START_POS
#define E2E_START_POS 0
#endif
constexpr uint32_t kMailboxTokens = 64;
static_assert(E2E_START_POS || E2E_PROMPT + E2E_NEW_TOKENS <= kMailboxTokens,
              "mailbox too small");
static_assert((E2E_START_POS ? E2E_START_POS : E2E_PROMPT) + E2E_NEW_TOKENS <=
                  kMaxContext,
              "KV cache too small");
static_assert(E2E_PROMPT + E2E_NEW_TOKENS <= kMaxContext, "KV cache too small");

// The host writes the prompt here before releasing the core and reads the
// generated tokens back. cycles[] holds mcycle at each phase boundary.
struct Mailbox {
    uint32_t prompt_length;
    uint32_t new_tokens;
    uint32_t status;
    uint32_t tokens[kMailboxTokens];
};
__attribute__((section(".mailbox"), used)) Mailbox mailbox = {
    E2E_START_POS ? E2E_START_POS : E2E_PROMPT, E2E_NEW_TOKENS, 0, {}};

Dtcm npu::dtcm;

namespace {

void layers(uint32_t tokens, uint32_t start_pos) {
    for (uint32_t layer = 0; layer < kNumLayers; ++layer) {
        const uint32_t base = kLayersBase + layer * kLayerBytes;
        layernorm(base + kNormOffset);
        mark(); // ln1
        linear(base + kQueryOffset, kHidden, kHidden, dtcm.normed, tokens,
               kFeatureMajor, dtcm.shared.attn.query);
        linear(base + kKeyOffset, kHidden, kHidden, dtcm.normed, tokens,
               kFeatureMajor, dtcm.shared.attn.key);
        linear(base + kValueOffset, kHidden, kHidden, dtcm.normed, tokens,
               kValueRows, reinterpret_cast<int8_t*>(dtcm.shared.attn.value));
        mark(); // qkv
        attention(layer, tokens, start_pos);
        mark(); // attention
        linear(base + kOutOffset, kHidden, kHidden, dtcm.normed, tokens,
               kResidual, dtcm.residual);
        mark(); // attn_out + residual
        layernorm(base + kNormOffset + 2 * kHidden);
        mark(); // ln2
        linear(base + kFc1Offset, kHidden, kMlpDim, dtcm.normed, tokens, kGelu,
               dtcm.shared.mlp);
        mark(); // fc1 + gelu
        linear(base + kFc2Offset, kMlpDim, kHidden, dtcm.shared.mlp, tokens,
               kResidual, dtcm.residual);
        mark(); // fc2 + residual
    }
}

// ln_f, then lm_head on token column token_col.
uint32_t next_token(uint32_t token_col) {
    layernorm(kFinalNorm);
    mark(); // ln_f
    const uint32_t token = lm_head(token_col);
    mark(); // lm_head
    return token;
}

} // namespace

int main() {
    init_tables();
    Mailbox& box = mailbox;
    mark(); // ROI start
    const uint32_t prompt = box.prompt_length;
    uint32_t next = 0;
    if (!E2E_START_POS) {
        uint32_t tokens = kBlockTokens;
        for (uint32_t start_pos = 0; start_pos < prompt;
             start_pos += kBlockTokens) {
            tokens = prompt - start_pos < kBlockTokens ? prompt - start_pos
                                                       : kBlockTokens;
            embed(box.tokens + start_pos, tokens, start_pos);
            mark(); // embedding
            layers(tokens, start_pos);
        }
        next = next_token(tokens - 1);
    }
    for (uint32_t step = 0; step < box.new_tokens; ++step) {
        const uint32_t pos = prompt + step;
        box.tokens[(E2E_START_POS ? step : pos) % kMailboxTokens] = next;
        if (E2E_START_POS || step + 1 < box.new_tokens) {
            embed(&next, 1, pos);
            mark(); // embedding
            layers(1, pos);
            next = next_token(0);
        }
    }
    box.status = 1;
    return 0;
}
