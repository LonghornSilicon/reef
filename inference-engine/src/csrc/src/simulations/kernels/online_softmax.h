// Online softmax for Coral NPU, built on quant_helpers.h.
#pragma once

#include <stddef.h>
#include <stdint.h>

// probs[j] = softmax(score_scale * scores)[j] for j in [lo, hi] (inclusive),
// over one int32 score row: the q . kt accumulator after the k_zero * q_sum
// fixup. Entries outside [lo, hi] are neither read nor written, so masked
// positions need no sentinel scores.
//
// Output is uint8 with scale 1/255 and zero point 0 (params::kProbsScale),
// the A operand matmul_vme_u8 takes for P . V. Each probability is rounded
// on its own, so a row sums to 255 only up to rounding (about +-(hi-lo)/2
// LSBs in the worst case).
//
// Two passes: the first keeps a running max and a sum of exp(s - max),
// rescaling the sum whenever the max grows (the online normalizer); the
// second writes the normalized, quantized probabilities.
extern "C" void online_softmax(const int32_t* scores, size_t lo, size_t hi,
                               float score_scale, uint8_t* probs);
