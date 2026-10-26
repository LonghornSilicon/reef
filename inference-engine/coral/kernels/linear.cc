// Linear projections: float, W8A8, int8, and the row-major matvec for
// lm_head.
//
// linear_f32 and linear_w8a8 produce up to 16 outputs per vector strip and
// walk the inputs in order, so each lane accumulates exactly like the
// reference's sequential sum. matvec_rows_f32 instead reduces along a row
// with the ordered reduction vfredosum, which adds in element order.
// linear_int8 is integer throughout, so its order is free.

#include "fixed_point.hpp"
#include "reef_kernels/kernels.hpp"

#include <riscv_vector.h>

namespace reef_kernels {
namespace {

constexpr float kInt8Limit = 127.0F;

// What happens to a strip of finished outputs before it is stored: nothing,
// or adding the bias (linear() adds it after the sum, as here).
struct NoBias {
    vfloat32m4_t operator()(vfloat32m4_t acc, std::size_t /*c*/,
                            std::size_t /*vl*/) const {
        return acc;
    }
};

struct AddBias {
    const float* bias;
    vfloat32m4_t operator()(vfloat32m4_t acc, std::size_t c,
                            std::size_t vl) const {
        return __riscv_vfadd_vv_f32m4(acc, __riscv_vle32_v_f32m4(bias + c, vl),
                                      vl);
    }
};

template <typename Epilogue>
void linear_f32_impl(std::size_t rows, std::size_t in, std::size_t out,
                     const float* x, const float* wt, Epilogue epilogue,
                     float* y) {
    for (std::size_t c = 0; c < out;) {
        const std::size_t vl = __riscv_vsetvl_e32m4(out - c);
        for (std::size_t r = 0; r < rows; ++r) {
            const float* xr = x + (r * in);
            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0F, vl);
            for (std::size_t k = 0; k < in; ++k) {
                const vfloat32m4_t w =
                    __riscv_vle32_v_f32m4(wt + (k * out) + c, vl);
                acc = __riscv_vfadd_vv_f32m4(
                    acc, __riscv_vfmul_vf_f32m4(w, xr[k], vl), vl);
            }
            __riscv_vse32_v_f32m4(y + (r * out) + c, epilogue(acc, c, vl), vl);
        }
        c += vl;
    }
}

// Quantize one row with its own scale (quantize_rows): returns the scale.
float quantize_row(std::size_t n, const float* x, std::int8_t* x_q) {
    vfloat32m1_t largest = __riscv_vfmv_s_f_f32m1(0.0F, 1);
    for (std::size_t k = 0; k < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m4(n - k);
        const vfloat32m4_t magnitude =
            __riscv_vfabs_v_f32m4(__riscv_vle32_v_f32m4(x + k, vl), vl);
        largest = __riscv_vfredmax_vs_f32m4_f32m1(magnitude, largest, vl);
        k += vl;
    }
    const float max_magnitude = __riscv_vfmv_f_s_f32m1_f32(largest);
    const float scale =
        max_magnitude > 0.0F ? max_magnitude / kInt8Limit : 1.0F;

    for (std::size_t k = 0; k < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m4(n - k);
        // x / scale, rounded to nearest even by the dynamic rounding mode.
        vint32m4_t q = __riscv_vfcvt_x_f_v_i32m4(
            __riscv_vfdiv_vf_f32m4(__riscv_vle32_v_f32m4(x + k, vl), scale, vl),
            vl);
        q = __riscv_vmax_vx_i32m4(q, -127, vl);
        q = __riscv_vmin_vx_i32m4(q, 127, vl);
        const vint16m2_t narrow = __riscv_vncvt_x_x_w_i16m2(q, vl);
        __riscv_vse8_v_i8m1(x_q + k, __riscv_vncvt_x_x_w_i8m1(narrow, vl), vl);
        k += vl;
    }
    return scale;
}

template <typename Epilogue>
void linear_w8a8_impl(std::size_t rows, std::size_t in, std::size_t out,
                      const float* x, const std::int8_t* wt,
                      const float* w_scale, Epilogue epilogue, std::int8_t* x_q,
                      float* y) {
    for (std::size_t r = 0; r < rows; ++r) {
        const float scale = quantize_row(in, x + (r * in), x_q);
        for (std::size_t c = 0; c < out;) {
            const std::size_t vl = __riscv_vsetvl_e32m4(out - c);
            vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
            for (std::size_t k = 0; k < in; ++k) {
                const vint16m2_t w = __riscv_vsext_vf2_i16m2(
                    __riscv_vle8_v_i8m1(wt + (k * out) + c, vl), vl);
                acc = __riscv_vwmacc_vx_i32m4(acc, x_q[k], w, vl);
            }
            // float(sum) * (row_scale * w_scale[c]), then the bias.
            const vfloat32m4_t combined = __riscv_vfmul_vf_f32m4(
                __riscv_vle32_v_f32m4(w_scale + c, vl), scale, vl);
            const vfloat32m4_t result = __riscv_vfmul_vv_f32m4(
                __riscv_vfcvt_f_x_v_f32m4(acc, vl), combined, vl);
            __riscv_vse32_v_f32m4(y + (r * out) + c, epilogue(result, c, vl),
                                  vl);
            c += vl;
        }
    }
}

// Where an int8 projection's int32 sums start: zero, or the bias.
struct ZeroStart {
    vint32m4_t operator()(std::size_t /*c*/, std::size_t vl) const {
        return __riscv_vmv_v_x_i32m4(0, vl);
    }
};

struct BiasStart {
    const std::int32_t* bias;
    vint32m4_t operator()(std::size_t c, std::size_t vl) const {
        return __riscv_vle32_v_i32m4(bias + c, vl);
    }
};

// 16 outputs per strip: acc[c] += x[r][k] * wt[k][c] over k, then narrow.
template <typename Start>
void linear_int8_impl(std::size_t rows, std::size_t in, std::size_t out,
                      const std::int8_t* x, const std::int8_t* wt, Start start,
                      unsigned shift, std::int8_t* y) {
    for (std::size_t c = 0; c < out;) {
        const std::size_t vl = __riscv_vsetvl_e8m1(out - c);
        for (std::size_t r = 0; r < rows; ++r) {
            vint32m4_t acc = start(c, vl);
            for (std::size_t k = 0; k < in; ++k) {
                const vint16m2_t w = __riscv_vsext_vf2_i16m2(
                    __riscv_vle8_v_i8m1(wt + (k * out) + c, vl), vl);
                acc = __riscv_vwmacc_vx_i32m4(acc, x[(r * in) + k], w, vl);
            }
            __riscv_vse8_v_i8m1(y + (r * out) + c,
                                detail::narrow_int8(acc, shift, vl), vl);
        }
        c += vl;
    }
}

} // namespace

void linear_f32(std::size_t rows, std::size_t in, std::size_t out,
                const float* x, const float* wt, float* y) {
    linear_f32_impl(rows, in, out, x, wt, NoBias{}, y);
}

void linear_bias_f32(std::size_t rows, std::size_t in, std::size_t out,
                     const float* x, const float* wt, const float* bias,
                     float* y) {
    linear_f32_impl(rows, in, out, x, wt, AddBias{bias}, y);
}

void linear_w8a8(std::size_t rows, std::size_t in, std::size_t out,
                 const float* x, const std::int8_t* wt, const float* w_scale,
                 std::int8_t* x_q, float* y) {
    linear_w8a8_impl(rows, in, out, x, wt, w_scale, NoBias{}, x_q, y);
}

void linear_w8a8_bias(std::size_t rows, std::size_t in, std::size_t out,
                      const float* x, const std::int8_t* wt,
                      const float* w_scale, const float* bias, std::int8_t* x_q,
                      float* y) {
    linear_w8a8_impl(rows, in, out, x, wt, w_scale, AddBias{bias}, x_q, y);
}

void matvec_rows_f32(std::size_t out, std::size_t in, const float* w,
                     const float* x, float* y) {
    for (std::size_t c = 0; c < out; ++c) {
        const float* row = w + (c * in);
        vfloat32m1_t sum = __riscv_vfmv_s_f_f32m1(0.0F, 1);
        for (std::size_t k = 0; k < in;) {
            const std::size_t vl = __riscv_vsetvl_e32m4(in - k);
            const vfloat32m4_t products =
                __riscv_vfmul_vv_f32m4(__riscv_vle32_v_f32m4(x + k, vl),
                                       __riscv_vle32_v_f32m4(row + k, vl), vl);
            sum = __riscv_vfredosum_vs_f32m4_f32m1(products, sum, vl);
            k += vl;
        }
        y[c] = __riscv_vfmv_f_s_f32m1_f32(sum);
    }
}

void linear_int8(std::size_t rows, std::size_t in, std::size_t out,
                 const std::int8_t* x, const std::int8_t* wt, unsigned shift,
                 std::int8_t* y) {
    linear_int8_impl(rows, in, out, x, wt, ZeroStart{}, shift, y);
}

void linear_bias_int8(std::size_t rows, std::size_t in, std::size_t out,
                      const std::int8_t* x, const std::int8_t* wt,
                      const std::int32_t* bias, unsigned shift,
                      std::int8_t* y) {
    linear_int8_impl(rows, in, out, x, wt, BiasStart{bias}, shift, y);
}

} // namespace reef_kernels
