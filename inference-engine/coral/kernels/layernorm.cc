// LayerNorm over rows (operator/layernorm.hpp).
//
// The reference sums the row in order for the mean and again for the
// variance; vfredosum adds in element order, so the sums match. The rest is
// one scalar square root and divide per row and elementwise operations.
// layernorm_int8 is the integer LayerNorm of int8.hpp.

#include "fixed_point.hpp"
#include "reef_kernels/kernels.hpp"

#include <riscv_vector.h>

namespace reef_kernels {
namespace {

// isqrt() and floor_div() from int8.hpp.
std::uint32_t isqrt(std::uint32_t value) {
    std::uint32_t root = 0;
    for (std::uint32_t bit = 1U << 30U; bit != 0; bit >>= 2U) {
        if (value >= root + bit) {
            value -= root + bit;
            root = (root >> 1U) + bit;
        } else {
            root >>= 1U;
        }
    }
    return root;
}

std::int32_t floor_div(std::int32_t numerator, std::int32_t denominator) {
    const std::int32_t quotient = numerator / denominator;
    return (numerator % denominator != 0 && numerator < 0) ? quotient - 1
                                                           : quotient;
}

vint32m4_t load_widened(const std::int8_t* p, std::size_t vl) {
    return __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(p, vl), vl);
}

} // namespace

void layernorm_f32(std::size_t rows, std::size_t width, const float* x,
                   const float* gain, const float* shift, float eps, float* y) {
    const auto count = static_cast<float>(width);
    for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x + (r * width);
        float* yr = y + (r * width);

        vfloat32m1_t sum = __riscv_vfmv_s_f_f32m1(0.0F, 1);
        for (std::size_t k = 0; k < width;) {
            const std::size_t vl = __riscv_vsetvl_e32m4(width - k);
            sum = __riscv_vfredosum_vs_f32m4_f32m1(
                __riscv_vle32_v_f32m4(xr + k, vl), sum, vl);
            k += vl;
        }
        const float mean = __riscv_vfmv_f_s_f32m1_f32(sum) / count;

        vfloat32m1_t squares = __riscv_vfmv_s_f_f32m1(0.0F, 1);
        for (std::size_t k = 0; k < width;) {
            const std::size_t vl = __riscv_vsetvl_e32m4(width - k);
            const vfloat32m4_t centred = __riscv_vfsub_vf_f32m4(
                __riscv_vle32_v_f32m4(xr + k, vl), mean, vl);
            squares = __riscv_vfredosum_vs_f32m4_f32m1(
                __riscv_vfmul_vv_f32m4(centred, centred, vl), squares, vl);
            k += vl;
        }
        const float variance = __riscv_vfmv_f_s_f32m1_f32(squares) / count;
        const float inv_std = 1.0F / __builtin_sqrtf(variance + eps);

        // ((x - mean) * inv_std) * gain + shift
        for (std::size_t k = 0; k < width;) {
            const std::size_t vl = __riscv_vsetvl_e32m4(width - k);
            vfloat32m4_t value = __riscv_vfsub_vf_f32m4(
                __riscv_vle32_v_f32m4(xr + k, vl), mean, vl);
            value = __riscv_vfmul_vf_f32m4(value, inv_std, vl);
            value = __riscv_vfmul_vv_f32m4(
                value, __riscv_vle32_v_f32m4(gain + k, vl), vl);
            value = __riscv_vfadd_vv_f32m4(
                value, __riscv_vle32_v_f32m4(shift + k, vl), vl);
            __riscv_vse32_v_f32m4(yr + k, value, vl);
            k += vl;
        }
    }
}

void layernorm_int8(std::size_t rows, std::size_t width, const std::int8_t* x,
                    const std::int8_t* gain, const std::int8_t* shift,
                    std::int8_t* y) {
    // Output scale 1/16 and Q5 gains: kLayerNormOutputBits and
    // kLayerNormGainBits in int8.hpp.
    constexpr std::int32_t kInvOne = 1 << (16 + 4);
    constexpr unsigned kGainBits = 5;
    const auto count = static_cast<std::int32_t>(width);
    for (std::size_t r = 0; r < rows; ++r) {
        const std::int8_t* xr = x + (r * width);
        std::int8_t* yr = y + (r * width);

        vint32m1_t sum = __riscv_vmv_s_x_i32m1(0, 1);
        for (std::size_t k = 0; k < width;) {
            const std::size_t vl = __riscv_vsetvl_e8m1(width - k);
            sum = __riscv_vredsum_vs_i32m4_i32m1(load_widened(xr + k, vl), sum,
                                                 vl);
            k += vl;
        }
        const std::int32_t mean =
            floor_div(__riscv_vmv_x_s_i32m1_i32(sum), count);

        vint32m1_t squares = __riscv_vmv_s_x_i32m1(0, 1);
        for (std::size_t k = 0; k < width;) {
            const std::size_t vl = __riscv_vsetvl_e8m1(width - k);
            const vint32m4_t centred =
                __riscv_vsub_vx_i32m4(load_widened(xr + k, vl), mean, vl);
            squares = __riscv_vredsum_vs_i32m4_i32m1(
                __riscv_vmul_vv_i32m4(centred, centred, vl), squares, vl);
            k += vl;
        }
        const std::uint32_t deviation = isqrt(static_cast<std::uint32_t>(
            __riscv_vmv_x_s_i32m1_i32(squares) / count));
        const std::int32_t inv =
            kInvOne / static_cast<std::int32_t>(deviation != 0 ? deviation : 1);

        // y = saturate(rounding_shift(((x - mean) * inv >> 16) * gain
        //                             + shift * 32, 5))
        for (std::size_t k = 0; k < width;) {
            const std::size_t vl = __riscv_vsetvl_e8m1(width - k);
            vint32m4_t value =
                __riscv_vsub_vx_i32m4(load_widened(xr + k, vl), mean, vl);
            value = __riscv_vsra_vx_i32m4(__riscv_vmul_vx_i32m4(value, inv, vl),
                                          16, vl);
            value =
                __riscv_vmul_vv_i32m4(value, load_widened(gain + k, vl), vl);
            value = __riscv_vadd_vv_i32m4(
                value,
                __riscv_vsll_vx_i32m4(load_widened(shift + k, vl), kGainBits,
                                      vl),
                vl);
            __riscv_vse8_v_i8m1(yr + k,
                                detail::narrow_int8(value, kGainBits, vl), vl);
            k += vl;
        }
    }
}

} // namespace reef_kernels
