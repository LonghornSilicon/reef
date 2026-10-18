// Lookup-table softmax for Coral NPU: integer max, Q15 exp from two tables,
// one division per row. See softmax_lut.h.
#include "softmax_lut.h"

#include "quant_helpers.h"

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
