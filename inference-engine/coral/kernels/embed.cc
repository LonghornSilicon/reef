// Token + position embeddings into the residual stream.

#include "common.hpp"

namespace npu {

void embed(const uint32_t* ids, uint32_t tokens, uint32_t start_pos) {
    for (uint32_t j = 0; j < tokens; ++j) {
        const uint32_t id = ids[j] < kVocab ? ids[j] : 0;
        dma(kTokenEmbed + id * kHidden, addr(dtcm.shared.embed_rows[j][0]),
            kHidden);
        dma(kPosEmbed + (start_pos + j) * kHidden,
            addr(dtcm.shared.embed_rows[j][1]), kHidden);
    }
    dma_wait();
    for (uint32_t j = 0; j < tokens; ++j) {
        for (uint32_t feature = 0; feature < kHidden; feature += 16) {
            __riscv_vsse8_v_i8m1(
                dtcm.residual + feature * kBlockTokens + j, kBlockTokens,
                __riscv_vsadd_vv_i8m1(
                    __riscv_vle8_v_i8m1(dtcm.shared.embed_rows[j][0] + feature,
                                        16),
                    __riscv_vle8_v_i8m1(dtcm.shared.embed_rows[j][1] + feature,
                                        16),
                    16),
                16);
        }
    }
}

} // namespace npu
