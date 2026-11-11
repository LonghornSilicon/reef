// LayerNorm over features, vectorized over the block's tokens. Integer only:
// out = centered * gamma / sqrt(var + 1) + beta, with 1/sqrt in Q15.

#include "common.hpp"

namespace npu {

// floor(sqrt(value)), bit by bit.
static uint32_t isqrt(uint32_t value) {
    uint32_t root = 0;
    for (uint32_t bit = 1U << 30; bit != 0; bit >>= 2) {
        if (value >= root + bit) {
            value -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
    }
    return root;
}

// ---- LayerNorm over features, vectorized over tokens: x -> nrm -----------
void layernorm(uint32_t params) {
    dma(params, addr(dtcm.shared.norm_params), 2 * kHidden);
    vint32m4_t sum = splat(0);
    vint32m4_t sum_sq = splat(0);
    for (uint32_t feature = 0; feature < kHidden; ++feature) {
        const vint32m4_t value =
            load_i8(dtcm.residual + feature * kBlockTokens);
        sum = __riscv_vadd_vv_i32m4(sum, value, 16);
        sum_sq = __riscv_vmacc_vv_i32m4(sum_sq, value, value, 16);
    }
    const vint32m4_t mean = __riscv_vdiv_vx_i32m4(sum, kHidden, 16);
    const vint32m4_t var = __riscv_vnmsac_vv_i32m4(
        __riscv_vdiv_vx_i32m4(sum_sq, kHidden, 16), mean, mean, 16);
    // 2^15 / sqrt(var + 1) per token: var + 1 is 1..16385, so 256..32768.
    __riscv_vse32_v_i32m4(dtcm.act_sum, __riscv_vadd_vx_i32m4(var, 1, 16), 16);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        dtcm.act_sum[lane] = static_cast<int32_t>(
            isqrt((1U << 30) / static_cast<uint32_t>(dtcm.act_sum[lane])));
    }
    const vint32m4_t rstd = __riscv_vle32_v_i32m4(dtcm.act_sum, 16);
    dma_wait();
    for (uint32_t feature = 0; feature < kHidden; ++feature) {
        const vint32m4_t centered = __riscv_vsub_vv_i32m4(
            load_i8(dtcm.residual + feature * kBlockTokens), mean, 16);
        // |centered * gamma| <= 255 * 128, times rstd <= 2^15: fits int32.
        vint32m4_t out = __riscv_vmul_vv_i32m4(
            __riscv_vmul_vx_i32m4(centered, dtcm.shared.norm_params[feature],
                                  16),
            rstd, 16);
        out = __riscv_vsra_vx_i32m4(__riscv_vadd_vx_i32m4(out, 1 << 14, 16), 15,
                                    16);
        out = __riscv_vadd_vx_i32m4(
            out, dtcm.shared.norm_params[kHidden + feature], 16);
        __riscv_vse8_v_i8m1(dtcm.normed + feature * kBlockTokens, to_i8(out),
                            kBlockTokens);
    }
}

} // namespace npu
