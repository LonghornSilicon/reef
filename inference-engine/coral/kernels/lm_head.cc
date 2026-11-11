// Tied lm_head over the padded vocabulary, greedy argmax.

#include "common.hpp"

namespace npu {

uint32_t lm_head(uint32_t token_col) {
    __riscv_vse32_v_i32m4(dtcm.best_logit, splat(kNegInf), 16);
    linear(kLmHead, kHidden, kVocabPadded, dtcm.normed + token_col, 1, kArgmax,
           nullptr);
    const vint32m4_t best = __riscv_vle32_v_i32m4(dtcm.best_logit, 16);
    const int32_t max_logit =
        __riscv_vmv_x_s_i32m1_i32(__riscv_vredmax_vs_i32m4_i32m1(
            best, __riscv_vmv_s_x_i32m1(kNegInf, 1), 16));
    const long lane =
        __riscv_vfirst_m_b8(__riscv_vmseq_vx_i32m4_b8(best, max_logit, 16), 16);
    return static_cast<uint32_t>(dtcm.best_index[lane < 0 ? 0 : lane]);
}

} // namespace npu
