// GELU and the exp/tanh it is built on, elementwise; the int8 GELU and
// softmax tables.

#include "reef_kernels/kernels.hpp"
#include "vector_math.hpp"

#include <riscv_vector.h>

namespace reef_kernels {
namespace {

// e = table[min((m - s) >> shift, 255)], widened to uint32.
vuint32m4_t exp_lookup(const std::int32_t* s, std::int32_t m, unsigned shift,
                       const std::uint16_t* table, std::size_t vl) {
    constexpr unsigned kLastEntry = 255;
    const vint32m4_t d = __riscv_vsra_vx_i32m4(
        __riscv_vrsub_vx_i32m4(__riscv_vle32_v_i32m4(s, vl), m, vl), shift, vl);
    vuint32m4_t index = __riscv_vminu_vx_u32m4(
        __riscv_vreinterpret_v_i32m4_u32m4(d), kLastEntry, vl);
    index = __riscv_vsll_vx_u32m4(index, 1, vl); // byte offset of a uint16
    return __riscv_vzext_vf2_u32m4(__riscv_vluxei32_v_u16m2(table, index, vl),
                                   vl);
}

} // namespace

void gelu_f32(std::size_t n, const float* x, float* y) {
    // Constants and evaluation order of gelu() in operator/activation.hpp:
    //   0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x * x * x)))
    constexpr float kCoefficient = 0.7978845608028654F;
    constexpr float kCubic = 0.044715F;
    for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m4(n - i);
        const vfloat32m4_t v = __riscv_vle32_v_f32m4(x + i, vl);
        vfloat32m4_t cubic = __riscv_vfmul_vf_f32m4(v, kCubic, vl);
        cubic = __riscv_vfmul_vv_f32m4(cubic, v, vl);
        cubic = __riscv_vfmul_vv_f32m4(cubic, v, vl);
        const vfloat32m4_t inner = __riscv_vfmul_vf_f32m4(
            __riscv_vfadd_vv_f32m4(v, cubic, vl), kCoefficient, vl);
        const vfloat32m4_t half = __riscv_vfmul_vf_f32m4(v, 0.5F, vl);
        const vfloat32m4_t one_plus_tanh =
            __riscv_vfadd_vf_f32m4(detail::tanh(inner, vl), 1.0F, vl);
        __riscv_vse32_v_f32m4(
            y + i, __riscv_vfmul_vv_f32m4(half, one_plus_tanh, vl), vl);
        i += vl;
    }
}

void exp_nonpositive_f32(std::size_t n, const float* x, float* y) {
    for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m4(n - i);
        __riscv_vse32_v_f32m4(
            y + i,
            detail::exp_nonpositive(__riscv_vle32_v_f32m4(x + i, vl), vl), vl);
        i += vl;
    }
}

void tanh_f32(std::size_t n, const float* x, float* y) {
    for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m4(n - i);
        __riscv_vse32_v_f32m4(
            y + i, detail::tanh(__riscv_vle32_v_f32m4(x + i, vl), vl), vl);
        i += vl;
    }
}

void gelu_int8(std::size_t n, const std::int8_t* x, const std::int8_t* table,
               std::int8_t* y) {
    constexpr unsigned kSignBit = 0x80;
    for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e8m1(n - i);
        // Flipping the sign bit turns x into the unsigned index x + 128.
        const vuint8m1_t index = __riscv_vxor_vx_u8m1(
            __riscv_vreinterpret_v_i8m1_u8m1(__riscv_vle8_v_i8m1(x + i, vl)),
            kSignBit, vl);
        __riscv_vse8_v_i8m1(y + i, __riscv_vluxei8_v_i8m1(table, index, vl),
                            vl);
        i += vl;
    }
}

// p[j] = round(255 * e[j] / sum(e)): row max, exp sum, one divide, scale.
void softmax_int8(std::size_t n, const std::int32_t* scores, unsigned shift,
                  const std::uint16_t* table, std::uint8_t* probs) {
    constexpr std::uint32_t kProbabilityOne = 255U << 16U;
    vint32m1_t largest = __riscv_vmv_s_x_i32m1(scores[0], 1);
    for (std::size_t j = 0; j < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m4(n - j);
        largest = __riscv_vredmax_vs_i32m4_i32m1(
            __riscv_vle32_v_i32m4(scores + j, vl), largest, vl);
        j += vl;
    }
    const std::int32_t m = __riscv_vmv_x_s_i32m1_i32(largest);
    vuint32m1_t sum = __riscv_vmv_s_x_u32m1(0, 1);
    for (std::size_t j = 0; j < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m4(n - j);
        sum = __riscv_vredsum_vs_u32m4_u32m1(
            exp_lookup(scores + j, m, shift, table, vl), sum, vl);
        j += vl;
    }
    // The maximum's weight is 32768, so the sum is never zero.
    const std::uint32_t reciprocal =
        kProbabilityOne / __riscv_vmv_x_s_u32m1_u32(sum);
    for (std::size_t j = 0; j < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m4(n - j);
        const vuint32m4_t scaled = __riscv_vmul_vx_u32m4(
            exp_lookup(scores + j, m, shift, table, vl), reciprocal, vl);
        const vuint16m2_t half =
            __riscv_vnclipu_wx_u16m2(scaled, 16, __RISCV_VXRM_RNU, vl);
        __riscv_vse8_v_u8m1(
            probs + j, __riscv_vnclipu_wx_u8m1(half, 0, __RISCV_VXRM_RNU, vl),
            vl);
        j += vl;
    }
}

} // namespace reef_kernels
