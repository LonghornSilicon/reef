// Softmax and GeLU for Coral NPU, both from lookup tables built once per
// layer. Header-only; built on quant_helpers.h.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "quant_helpers.h"

// ---- Softmax ----------------------------------------------------------------
// exp from two 256-entry tables: no floating point and no expf per score.

// Entries softmax_lut_table writes: 256 coarse, then 256 fine.
constexpr size_t kSoftmaxLutEntries = 512;

// Fills table[512] with exp(-score_scale * d) in Q15 (1.0 = 32768) for a
// score gap d = max - s split into its high and low bytes:
//   table[i]       = exp(-score_scale * 256 * i)
//   table[256 + j] = exp(-score_scale * j)
// so exp(-score_scale * d) = table[d >> 8] * table[256 + (d & 255)] is
// exact up to the Q15 rounding. Gaps of 65536 or more read as 0, which
// needs score_scale * 65536 >= 13 (exp(-13) is under 1/255 / 1000);
// GPT-2's score scale gives 20.5. Build it once per score_scale, i.e. once
// per layer, like gelu_table.
REEF_KERNEL void softmax_lut_table(float score_scale, uint16_t* table) {
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

// exp(-score_scale * d) in Q15 for a score gap d >= 0.
inline uint32_t exp_lut(uint32_t d, const uint16_t* table) {
    if (d >= 65536U) {
        return 0;
    }
    const uint32_t coarse = table[d >> 8U];
    const uint32_t fine = table[256U + (d & 255U)];
    return ((coarse * fine) + (1U << 14U)) >> 15U;
}

// probs[j] for j in [lo, hi], uint8 with scale 1/255, nothing outside the
// range touched, computed in integers: one pass for the int32 max, one for
// the Q15 sum through exp_lut, and one to normalize with a single division
// per row. Within one 1/255 step of the exact softmax.
REEF_KERNEL void softmax_lut(const int32_t* scores, size_t lo, size_t hi,
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

// ---- GeLU -------------------------------------------------------------------
// A 256-entry lookup table, so it is exact for this quantization: every int8
// input has its own precomputed output.

// Fills table[256] so that table[x + 128] = GeLU(x * in_scale) requantized
// to int8 with out_scale (both zero point 0). GeLU is GPT-2's tanh form:
// 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). Build it once per pair
// of scales.
REEF_KERNEL void gelu_table(float in_scale, float out_scale, int8_t* table) {
    constexpr float kSqrt2OverPi = 0.7978845608F;
    for (int32_t i = 0; i < 256; ++i) {
        const float x = static_cast<float>(i - 128) * in_scale;
        const float inner = kSqrt2OverPi * (x + (0.044715F * x * x * x));
        table[i] = quant::to_int8(0.5F * x * (1.0F + tanhf(inner)), out_scale);
    }
}

// y[i] = table[x[i] + 128] for i < n.
REEF_KERNEL void gelu(size_t n, const int8_t* x, const int8_t* table,
                      int8_t* y) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = table[x[i] + 128];
    }
}
