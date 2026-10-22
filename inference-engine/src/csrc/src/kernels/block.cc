// Transformer-block glue for Coral NPU that has no engine operator yet.
// See kernels.h.
//   layernorm: exact int32 row stats, float normalize
//   residual_add: dequantize both inputs, add, requantize
#include "helpers.h"
#include "kernels.h"

namespace {
constexpr float kLayernormEps = 1e-5F; // GPT-2's
} // namespace

extern "C" void layernorm(size_t rows, size_t d, const int8_t* x, float x_scale,
                          const int8_t* gamma, float gamma_scale,
                          const int8_t* beta, float beta_scale, float out_scale,
                          int8_t* out) {
    const float inv_d = 1.0F / static_cast<float>(d);
    const auto d_i = static_cast<int32_t>(d);
    size_t vl = 0;
    for (size_t r = 0; r < rows; ++r) {
        const int8_t* row = x + (r * d);
        int32_t sum = 0;
        int32_t sum_sq = 0;
        for (size_t c = 0; c < d; c += vl) {
            vl = rvv::set_vl(d - c);
            const rvv::vi8 v = rvv::load_i8(row + c, vl);
            sum += rvv::sum(rvv::widen(v, vl), vl);
            sum_sq += rvv::sum_squares(v, vl);
        }
        // stats in x's integer units; eps moves into them via x_scale^2
        const float mean = static_cast<float>(sum) * inv_d;
        const float var = (static_cast<float>(sum_sq) * inv_d) - (mean * mean);
        const float eps = kLayernormEps / (x_scale * x_scale);
        const float inv_std = 1.0F / sqrtf(var + eps);
        // out = (d x - sum) gamma * f1 + beta * f2, in fixed point: the
        // first term through a Q31 multiply, the second as beta * f2 * 2^s
        // in the same units, then one rounding shift by s
        const quant::Requant f1 =
            quant::requant_params(inv_std * gamma_scale * inv_d / out_scale);
        const int32_t f2 = quant::to_fixed(beta_scale / out_scale, f1.shift);
        for (size_t c = 0; c < d; c += vl) {
            vl = rvv::set_vl(d - c);
            rvv::vi32 t = rvv::add_x(
                rvv::mul_x(rvv::widen(rvv::load_i8(row + c, vl), vl), d_i, vl),
                -sum, vl);
            t = rvv::mul(t, rvv::widen(rvv::load_i8(gamma + c, vl), vl), vl);
            t = rvv::q31_mul(t, f1.multiplier, vl);
            t = rvv::mac_x(t, f2, rvv::widen(rvv::load_i8(beta + c, vl), vl),
                           vl);
            rvv::store_i8(out + (r * d) + c,
                          rvv::round_shift_to_int8(t, f1.shift, vl), vl);
        }
    }
}

extern "C" void residual_add(size_t n, const int8_t* a, float a_scale,
                             const int8_t* b, float b_scale, float out_scale,
                             int8_t* out) {
    // a * fa + b * fb with both factors as q-bit fixed point
    const float fa = a_scale / out_scale;
    const float fb = b_scale / out_scale;
    const int32_t q = quant::fraction_bits(fa + fb);
    const int32_t fa_q = quant::to_fixed(fa, q);
    const int32_t fb_q = quant::to_fixed(fb, q);
    size_t vl = 0;
    for (size_t i = 0; i < n; i += vl) {
        vl = rvv::set_vl(n - i);
        rvv::vi32 t =
            rvv::mul_x(rvv::widen(rvv::load_i8(a + i, vl), vl), fa_q, vl);
        t = rvv::mac_x(t, fb_q, rvv::widen(rvv::load_i8(b + i, vl), vl), vl);
        rvv::store_i8(out + i, rvv::round_shift_to_int8(t, q, vl), vl);
    }
}
