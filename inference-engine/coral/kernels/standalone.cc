// Runs one kernel once on one block of E2E_TOKENS tokens (1..kBlockTokens,
// default kBlockTokens) of layer E2E_LAYER (default 0), with the block's
// residual already in the DTCM. E2E_START_POS sets the block's first position
// (attention reads that many cached positions). E2E_TOKENS=1 is one decode
// step. Build with -DKERNEL_<NAME> (build.sh does). The two mcycle reads
// around the kernel are the ROI reef-perf reports.

#include "common.hpp"

using namespace npu;

#ifndef E2E_START_POS
#define E2E_START_POS 0
#endif
#ifndef E2E_TOKENS
#define E2E_TOKENS kBlockTokens
#endif
#ifndef E2E_LAYER
#define E2E_LAYER 0
#endif
static_assert(E2E_TOKENS >= 1 && E2E_TOKENS <= kBlockTokens, "E2E_TOKENS");
static_assert(E2E_LAYER < kNumLayers, "E2E_LAYER");
static_assert(E2E_START_POS + E2E_TOKENS <= kMaxPositions &&
                  E2E_START_POS + E2E_TOKENS <= kMaxContext,
              "E2E_START_POS + E2E_TOKENS past the position table or KV cache");

Dtcm npu::dtcm;

int main() {
    init_tables();
    [[maybe_unused]] constexpr uint32_t kTokens = E2E_TOKENS;
    [[maybe_unused]] constexpr uint32_t kPos = E2E_START_POS;
    [[maybe_unused]] constexpr uint32_t kLayer =
        kLayersBase + E2E_LAYER * kLayerBytes; // the layer's weights
    mark();
#if defined(KERNEL_EMBED)
    static const uint32_t ids[kBlockTokens] = {};
    embed(ids, kTokens, kPos);
#elif defined(KERNEL_LAYERNORM)
    layernorm(kLayer + kNormOffset);
#elif defined(KERNEL_QKV)
    linear(kLayer + kQueryOffset, kHidden, kHidden, dtcm.normed, kTokens,
           kFeatureMajor, dtcm.shared.attn.query);
    linear(kLayer + kKeyOffset, kHidden, kHidden, dtcm.normed, kTokens,
           kFeatureMajor, dtcm.shared.attn.key);
    linear(kLayer + kValueOffset, kHidden, kHidden, dtcm.normed, kTokens,
           kValueRows, reinterpret_cast<int8_t*>(dtcm.shared.attn.value));
#elif defined(KERNEL_ATTENTION)
    attention(E2E_LAYER, kTokens, kPos);
#elif defined(KERNEL_ATTN_OUT)
    linear(kLayer + kOutOffset, kHidden, kHidden, dtcm.normed, kTokens,
           kResidual, dtcm.residual);
#elif defined(KERNEL_FC1)
    linear(kLayer + kFc1Offset, kHidden, kMlpDim, dtcm.normed, kTokens, kGelu,
           dtcm.shared.mlp);
#elif defined(KERNEL_FC2)
    linear(kLayer + kFc2Offset, kMlpDim, kHidden, dtcm.shared.mlp, kTokens,
           kResidual, dtcm.residual);
#elif defined(KERNEL_LM_HEAD)
    (void)lm_head(kTokens - 1);
#else
#error "build with -DKERNEL_<NAME>"
#endif
    mark();
    return 0;
}
