#include "mlp_w8a8.h"
#include <math.h>

namespace {
int8_t round_even(float value) {
    const float lower = floorf(value);
    int rounded = static_cast<int>(lower);
    const float fraction = value - lower;
    if (fraction > 0.5f || (fraction == 0.5f && rounded % 2 != 0)) {
        ++rounded;
    }
    if (rounded > 127) rounded = 127;
    if (rounded < -127) rounded = -127;
    return static_cast<int8_t>(rounded);
}
} // namespace

bool linear_w8a8(size_t tokens, size_t in, size_t out, const int8_t* x,
                 const float* x_scale, const int8_t* w, const float* w_scale,
                 const float* bias, int32_t* accumulators, float* output) {
    // Safe even for signed -128 operands: 131072 * 128 * 128 can overflow,
    // so cap below that boundary. Generated symmetric fixtures use ±127.
    if (tokens == 0 || in == 0 || out == 0 || in >= 131072) return false;
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t row = 0; row < out; ++row) {
            int32_t sum = 0;
            for (size_t col = 0; col < in; ++col) {
                sum += static_cast<int32_t>(x[token * in + col]) *
                       static_cast<int32_t>(w[row * in + col]);
            }
            accumulators[token * out + row] = sum;
            output[token * out + row] =
                (static_cast<float>(sum) * x_scale[token]) * w_scale[row] +
                bias[row];
        }
    }
    return true;
}

void quantize_tokens(size_t tokens, size_t width, const float* input,
                     int8_t* output, float* scales) {
    for (size_t token = 0; token < tokens; ++token) {
        float maximum = 0.0f;
        for (size_t col = 0; col < width; ++col) {
            maximum = fmaxf(maximum, fabsf(input[token * width + col]));
        }
        scales[token] = maximum == 0.0f ? 1.0f : maximum / 127.0f;
        for (size_t col = 0; col < width; ++col) {
            output[token * width + col] =
                round_even(input[token * width + col] / scales[token]);
        }
    }
}

bool mlp_w8a8(size_t tokens, size_t width, size_t hidden, const int8_t* x,
              const float* x_scale, const int8_t* w1, const float* s1,
              const float* b1, const int8_t* w2, const float* s2,
              const float* b2, int32_t* a1, float* hidden_fp32,
              int8_t* hidden_q, float* hidden_scale, int32_t* a2,
              float* output) {
    if (width == 0 || hidden == 0 || width >= 131072 || hidden >= 131072)
        return false;
    if (!linear_w8a8(tokens, width, hidden, x, x_scale, w1, s1, b1, a1,
                     hidden_fp32)) return false;
    for (size_t i = 0; i < tokens * hidden; ++i) {
        const float xh = hidden_fp32[i];
        hidden_fp32[i] = 0.5f * xh *
            (1.0f + tanhf(0.7978845608028654f *
                          (xh + 0.044715f * xh * xh * xh)));
    }
    quantize_tokens(tokens, hidden, hidden_fp32, hidden_q, hidden_scale);
    return linear_w8a8(tokens, hidden, width, hidden_q, hidden_scale,
                       w2, s2, b2, a2, output);
}
