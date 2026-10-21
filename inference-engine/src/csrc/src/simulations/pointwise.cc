// Pointwise ops for Coral NPU. See kernels.h.
//   gelu: build a 256-entry table once, then look up
//   layernorm: exact int32 row stats, float normalize
//   residual_add: dequantize both inputs, add, requantize
#include "helpers.h"
#include "kernels.h"

namespace {
constexpr float kLayernormEps = 1e-5F; // GPT-2's
} // namespace

extern "C" void gelu_table(float in_scale, float out_scale, int8_t* table) {
    constexpr float kSqrt2OverPi = 0.7978845608F;
    for (int32_t i = 0; i < 256; ++i) {
        const float x = static_cast<float>(i - 128) * in_scale;
        const float inner = kSqrt2OverPi * (x + (0.044715F * x * x * x));
        table[i] = quant::to_int8(0.5F * x * (1.0F + tanhf(inner)), out_scale);
    }
}

extern "C" void gelu(size_t n, const int8_t* x, const int8_t* table,
                     int8_t* y) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = table[x[i] + 128];
    }
}

extern "C" void layernorm(size_t rows, size_t d, const int8_t* x, float x_scale,
                          const int8_t* gamma, float gamma_scale,
                          const int8_t* beta, float beta_scale, float out_scale,
                          int8_t* out) {
    const float inv_d = 1.0F / static_cast<float>(d);
    for (size_t r = 0; r < rows; ++r) {
        const int8_t* row = x + (r * d);
        int32_t sum = 0;
        int32_t sum_sq = 0;
        for (size_t c = 0; c < d; ++c) {
            sum += row[c];
            sum_sq += row[c] * row[c];
        }
        // stats in x's integer units; eps moves into them via x_scale^2
        const float mean = static_cast<float>(sum) * inv_d;
        const float var = (static_cast<float>(sum_sq) * inv_d) - (mean * mean);
        const float eps = kLayernormEps / (x_scale * x_scale);
        const float inv_std = 1.0F / sqrtf(var + eps);
        for (size_t c = 0; c < d; ++c) {
            const float norm = (static_cast<float>(row[c]) - mean) * inv_std;
            const float value =
                (norm * gamma_scale * static_cast<float>(gamma[c])) +
                (beta_scale * static_cast<float>(beta[c]));
            out[(r * d) + c] = quant::to_int8(value, out_scale);
        }
    }
}

extern "C" void residual_add(size_t n, const int8_t* a, float a_scale,
                             const int8_t* b, float b_scale, float out_scale,
                             int8_t* out) {
    for (size_t i = 0; i < n; ++i) {
        const float value = (a_scale * static_cast<float>(a[i])) +
                            (b_scale * static_cast<float>(b[i]));
        out[i] = quant::to_int8(value, out_scale);
    }
}
