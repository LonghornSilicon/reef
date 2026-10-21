// The activation operator (operator/activation.hpp) for Coral NPU. See
// kernels.h.
//   gelu: build a 256-entry table once, then look up
//   softmax_lut: integer max, Q15 exp from two tables, one division per row;
//   its [lo, hi] range is masked_softmax's mask
#include "helpers.h"
#include "kernels.h"

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

extern "C" void softmax_lut_table(float score_scale, uint16_t* table) {
    for (size_t i = 0; i < 256; ++i) {
        const auto gap = static_cast<float>(i);
        // exp(-x) is in (0, 1], so adding 0.5 and truncating rounds
        // NOLINTBEGIN(bugprone-incorrect-roundings)
        table[i] = static_cast<uint16_t>(
            (32768.0F * expf(-score_scale * 256.0F * gap)) + 0.5F);
        table[256 + i] =
            static_cast<uint16_t>((32768.0F * expf(-score_scale * gap)) + 0.5F);
        // NOLINTEND(bugprone-incorrect-roundings)
    }
}

extern "C" void softmax_lut(const int32_t* scores, size_t lo, size_t hi,
                            const uint16_t* table, uint8_t* probs) {
    int32_t m = scores[lo];
    for (size_t j = lo + 1; j <= hi; ++j) {
        m = scores[j] > m ? scores[j] : m;
    }

    // l >= 32768 (the max scores 1.0), and fits 32 bits for rows under 2^17
    uint32_t l = 0;
    for (size_t j = lo; j <= hi; ++j) {
        l += exp_lut(static_cast<uint32_t>(m - scores[j]), table);
    }

    // p = 255 e / l as e * (255 * 2^24 / l) / 2^24; e <= l keeps the
    // product under 2^32. l > 0 since the max's exp_lut(0) is 32768.
    // NOLINTNEXTLINE(clang-analyzer-core.DivideZero)
    const uint32_t r = (255U << 24U) / l;
    for (size_t j = lo; j <= hi; ++j) {
        const uint32_t e = exp_lut(static_cast<uint32_t>(m - scores[j]), table);
        probs[j] = static_cast<uint8_t>(((e * r) + (1U << 23U)) >> 24U);
    }
}
