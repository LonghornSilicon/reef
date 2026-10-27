#pragma once

// Vector versions of exp_nonpositive() and tanh_float() (math.hpp). Each
// line performs the same IEEE-754 operation as the matching line there, so
// every lane rounds exactly as the host reference does. Keep the two in
// step: change math.hpp first, then this file.

#include <riscv_vector.h>

#include <cstddef>

namespace reef_kernels {
namespace detail {

// e^x for x <= 0, lane by lane; lanes below -87 become 0.
inline vfloat32m4_t exp_nonpositive(vfloat32m4_t x, std::size_t vl) {
    constexpr float kMinInput = -87.0F;
    constexpr float kLog2e = 1.44269504088896341F;
    constexpr float kRoundToInteger = 12582912.0F;
    constexpr float kLn2High = 0.693359375F;
    constexpr float kLn2Low = -2.12194440e-4F;
    constexpr int kExponentBias = 127;
    constexpr int kMantissaBits = 23;

    const vbool8_t underflow = __riscv_vmflt_vf_f32m4_b8(x, kMinInput, vl);
    vfloat32m4_t k = __riscv_vfmul_vf_f32m4(x, kLog2e, vl);
    k = __riscv_vfadd_vf_f32m4(k, kRoundToInteger, vl);
    k = __riscv_vfsub_vf_f32m4(k, kRoundToInteger, vl);
    vfloat32m4_t r =
        __riscv_vfsub_vv_f32m4(x, __riscv_vfmul_vf_f32m4(k, kLn2High, vl), vl);
    r = __riscv_vfsub_vv_f32m4(r, __riscv_vfmul_vf_f32m4(k, kLn2Low, vl), vl);
    const vfloat32m4_t r2 = __riscv_vfmul_vv_f32m4(r, r, vl);

    vfloat32m4_t poly = __riscv_vfmv_v_f_f32m4(1.9875691500E-4F, vl);
    poly = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(poly, r, vl),
                                  1.3981999507E-3F, vl);
    poly = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(poly, r, vl),
                                  8.3334519073E-3F, vl);
    poly = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(poly, r, vl),
                                  4.1665795894E-2F, vl);
    poly = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(poly, r, vl),
                                  1.6666665459E-1F, vl);
    poly = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(poly, r, vl),
                                  5.0000001201E-1F, vl);
    poly = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(poly, r2, vl), r, vl);
    poly = __riscv_vfadd_vf_f32m4(poly, 1.0F, vl);

    // 2^k from the exponent field. Underflowing lanes get a meaningless
    // scale here and are replaced by 0 below.
    vint32m4_t exponent = __riscv_vfcvt_rtz_x_f_v_i32m4(k, vl);
    exponent = __riscv_vadd_vx_i32m4(exponent, kExponentBias, vl);
    exponent = __riscv_vsll_vx_i32m4(exponent, kMantissaBits, vl);
    const vfloat32m4_t scale = __riscv_vreinterpret_v_i32m4_f32m4(exponent);
    const vfloat32m4_t y = __riscv_vfmul_vv_f32m4(poly, scale, vl);
    return __riscv_vfmerge_vfm_f32m4(y, 0.0F, underflow, vl);
}

// tanh(x) = sign(x) * (1 - e) / (1 + e) with e = exp(-2|x|).
inline vfloat32m4_t tanh(vfloat32m4_t x, std::size_t vl) {
    const vfloat32m4_t magnitude = __riscv_vfabs_v_f32m4(x, vl);
    const vfloat32m4_t e =
        exp_nonpositive(__riscv_vfmul_vf_f32m4(magnitude, -2.0F, vl), vl);
    const vfloat32m4_t t =
        __riscv_vfdiv_vv_f32m4(__riscv_vfrsub_vf_f32m4(e, 1.0F, vl),
                               __riscv_vfadd_vf_f32m4(e, 1.0F, vl), vl);
    return __riscv_vfsgnj_vv_f32m4(t, x, vl);
}

} // namespace detail
} // namespace reef_kernels
