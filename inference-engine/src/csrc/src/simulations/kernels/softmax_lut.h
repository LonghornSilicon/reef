// Softmax for Coral NPU with exp from two 256-entry lookup tables: no
// floating point and no expf per score.
#pragma once

#include <stddef.h>
#include <stdint.h>

// Entries softmax_lut_table writes: 256 coarse, then 256 fine.
constexpr size_t kSoftmaxLutEntries = 512;

// Fills table[512] with exp(-score_scale * d) in Q15 (1.0 = 32768) for a
// score gap d = max - s split into its high and low bytes:
//   table[i]       = exp(-score_scale * 256 * i)
//   table[256 + j] = exp(-score_scale * j)
// so exp(-score_scale * d) = table[d >> 8] * table[256 + (d & 255)] is
// exact up to the Q15 rounding. Gaps of 65536 or more read as 0, which
// needs score_scale * 65536 >= 13 (exp(-13) is under 1/255 / 1000);
// GPT-2's params::kScoreScale gives 20.5. Build it once per score_scale,
// i.e. once per layer, like gelu_table.
extern "C" void softmax_lut_table(float score_scale, uint16_t* table);

// exp(-score_scale * d) in Q15 for a score gap d >= 0.
inline uint32_t exp_lut(uint32_t d, const uint16_t* table) {
    if (d >= 65536U) {
        return 0;
    }
    const uint32_t coarse = table[d >> 8U];
    const uint32_t fine = table[256U + (d & 255U)];
    return ((coarse * fine) + (1U << 14U)) >> 15U;
}

// online_softmax's contract (probs[j] for j in [lo, hi], uint8 with scale
// 1/255, nothing outside the range touched), computed in integers: one pass
// for the int32 max, one for the Q15 sum through exp_lut, and one to
// normalize with a single division per row. Within one 1/255 step of the
// exact softmax, like online_softmax.
extern "C" void softmax_lut(const int32_t* scores, size_t lo, size_t hi,
                            const uint16_t* table, uint8_t* probs);
