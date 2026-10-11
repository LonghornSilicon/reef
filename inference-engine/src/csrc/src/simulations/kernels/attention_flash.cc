// Flash attention for Coral NPU: the fused counterpart of attention_serial.
//
// Each query row makes one pass over its visible keys [lo, hi], keeping a
// running maximum m, denominator l and weighted value sum acc[d]. When a
// key raises the maximum, the old state is rescaled by exp(m_old - m_new)
// instead of being recomputed: the online-softmax recurrence the flash
// algorithm is built on. The first visible key seeds m directly, so no
// sentinel scores exist here either.
//
// TODO: port the matmul kernels in after the standalone softmax kernel
// lands, as a separate block-flash kernel (this scalar one stays the
// reference it is checked against). VME emits whole 16 x 16 score tiles
// rather than one score at a time, so the online update runs once per
// block; see the matching TODO in attention_serial.cc for why the softmax
// kernel, VME verification and tile shape rules gate the port.
#include "attention_flash.h"

#include "quant_helpers.h"

namespace {

// Stack bound for one head; see the limits note in attention_serial.h.
constexpr size_t kMaxHeadDim = 128;

// int8 dot product of query row `row` and key column `j`, dequantized into
// a float score. q_sum carries the precomputed zero-point lift:
// q . (k - k_zero) = q . k - k_zero * sum(q).
float score_at(size_t row, size_t j, size_t nk, size_t d, const int8_t* q,
               const uint8_t* kt, int32_t k_zero, int32_t q_sum,
               float score_scale) {
    int32_t acc = 0;
    for (size_t c = 0; c < d; ++c) {
        acc += static_cast<int32_t>(q[(row * d) + c]) *
               static_cast<int32_t>(kt[(c * nk) + j]);
    }
    return score_scale * static_cast<float>(acc - (k_zero * q_sum));
}

} // namespace

extern "C" void attention_flash(size_t nq, size_t nk, size_t d,
                                size_t query_start, size_t window,
                                const int8_t* q, const uint8_t* kt,
                                int32_t k_zero, const uint8_t* v,
                                int32_t v_zero, float score_scale,
                                float v_scale, float out_scale, int8_t* out) {
    float acc[kMaxHeadDim];
    for (size_t row = 0; row < nq; ++row) {
        const size_t pos = query_start + row;
        const size_t lo = (pos + 1 > window) ? pos + 1 - window : 0;
        const size_t hi = pos; // inclusive; pos < nk per the cache invariant

        int32_t q_sum = 0;
        for (size_t c = 0; c < d; ++c) {
            q_sum += q[(row * d) + c];
        }

        // the first visible key seeds the running state: its probability
        // weight is exp(0) = 1 before anything rescales it
        float m = score_at(row, lo, nk, d, q, kt, k_zero, q_sum, score_scale);
        float l = 1.0F;
        for (size_t c = 0; c < d; ++c) {
            acc[c] = static_cast<float>(v[(lo * d) + c]);
        }

        // online pass over the remaining visible keys
        for (size_t j = lo + 1; j <= hi; ++j) {
            const float s =
                score_at(row, j, nk, d, q, kt, k_zero, q_sum, score_scale);
            const float m_new = s > m ? s : m;
            const float correction = expf(m - m_new);
            const float p = expf(s - m_new);
            l = (l * correction) + p;
            for (size_t c = 0; c < d; ++c) {
                acc[c] = (acc[c] * correction) +
                         (p * static_cast<float>(v[(j * d) + c]));
            }
            m = m_new;
        }

        // finish: normalize, lift v's zero point out, requantize
        for (size_t c = 0; c < d; ++c) {
            const float value =
                v_scale * ((acc[c] / l) - static_cast<float>(v_zero));
            out[(row * d) + c] = quant::to_int8(value, out_scale);
        }
    }
}
