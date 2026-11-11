// LayerNorm over features, vectorized over the block's tokens.

#include "common.hpp"

namespace npu {

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
    const vfloat32m4_t rstd = __riscv_vfrdiv_vf_f32m4(
        __riscv_vfsqrt_v_f32m4(
            __riscv_vfcvt_f_x_v_f32m4(__riscv_vadd_vx_i32m4(var, 1, 16), 16),
            16),
        32.0F, 16);
    dma_wait();
    for (uint32_t feature = 0; feature < kHidden; ++feature) {
        const vint32m4_t centered = __riscv_vsub_vv_i32m4(
            load_i8(dtcm.residual + feature * kBlockTokens), mean, 16);
        vfloat32m4_t out = __riscv_vfmul_vv_f32m4(
            __riscv_vfcvt_f_x_v_f32m4(centered, 16), rstd, 16);
        out = __riscv_vfmul_vf_f32m4(
            out,
            static_cast<float>(dtcm.shared.norm_params[feature]) * 0.03125F,
            16);
        out = __riscv_vfadd_vf_f32m4(
            out, static_cast<float>(dtcm.shared.norm_params[kHidden + feature]),
            16);
        __riscv_vse8_v_i8m1(dtcm.normed + feature * kBlockTokens,
                            to_i8(__riscv_vfcvt_x_f_v_i32m4(out, 16)),
                            kBlockTokens);
    }
}

} // namespace npu
