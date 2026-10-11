// Serial attention for Coral NPU: the unfused reference the flash kernel is
// checked against for accuracy and, later, cycle counts.
//
// Per query row: int8 dot products build a float score row, a softmax
// normalizes the visible slice, and the probabilities weight the value
// rows. Only the visible positions [lo, hi] are ever touched, so masked
// positions need no sentinel scores.
//
// TODO: port the matmul kernels in after the standalone softmax kernel
// lands, as a sibling VME kernel (this scalar one stays the reference it
// is checked against). The pieces have to arrive in that order because:
// 1. The score stage maps to matmul_vme_s8u8 (q int8 x kt uint8 -> int32)
//    plus a per-row k_zero * q_sum fixup, but the value stage needs uint8
//    probabilities to feed matmul_vme_u8 - and uint8 probabilities are the
//    standalone softmax kernel's output, which does not exist yet.
// 2. VME only runs on Coral's RTL model (and matmul_rvv has never run
//    anywhere), so a port today could not be verified.
// 3. VME's shape rules (n % 16, m % 16, k % 4; whole 16 x 16 tiles) need
//    padding and masked-position handling that decode (nq = 1) does not
//    naturally fit.
#include "attention_serial.h"

#include "quant_helpers.h"

namespace {

// Stack bounds for one head; see the limits note in the header.
constexpr size_t kMaxHeadDim = 128;
constexpr size_t kMaxKeys = 1024;

// Softmax over scores[lo..hi] in place, three passes: max, exp-sum,
// normalize.
// TODO: temporary private stand-in. Port the standalone softmax kernel in
// here once it lands (see unit_tests/test_softmax.cc); this function is
// the only seam that should need to change.
void softmax_row(float* scores, size_t lo, size_t hi) {
    float m = scores[lo];
    for (size_t j = lo + 1; j <= hi; ++j) {
        m = scores[j] > m ? scores[j] : m;
    }
    float sum = 0;
    for (size_t j = lo; j <= hi; ++j) {
        scores[j] = expf(scores[j] - m);
        sum += scores[j];
    }
    for (size_t j = lo; j <= hi; ++j) {
        scores[j] /= sum;
    }
}

} // namespace

extern "C" void attention_serial(size_t nq, size_t nk, size_t d,
                                 size_t query_start, size_t window,
                                 const int8_t* q, const uint8_t* kt,
                                 int32_t k_zero, const uint8_t* v,
                                 int32_t v_zero, float score_scale,
                                 float v_scale, float out_scale, int8_t* out) {
    float scores[kMaxKeys];
    float mixed[kMaxHeadDim];
    for (size_t row = 0; row < nq; ++row) {
        const size_t pos = query_start + row;
        const size_t lo = (pos + 1 > window) ? pos + 1 - window : 0;
        const size_t hi = pos; // inclusive; pos < nk per the cache invariant

        // zero-point lift: q . (k - k_zero) = q . k - k_zero * sum(q)
        int32_t q_sum = 0;
        for (size_t c = 0; c < d; ++c) {
            q_sum += q[(row * d) + c];
        }

        // int8 dot products, dequantized into float scores
        for (size_t j = lo; j <= hi; ++j) {
            int32_t acc = 0;
            for (size_t c = 0; c < d; ++c) {
                acc += static_cast<int32_t>(q[(row * d) + c]) *
                       static_cast<int32_t>(kt[(c * nk) + j]);
            }
            const int32_t corrected = acc - (k_zero * q_sum);
            scores[j] = score_scale * static_cast<float>(corrected);
        }

        softmax_row(&scores[0], lo, hi);

        // probability-weighted value sum; v's zero point lifts out of the
        // sum because the probabilities add up to one
        for (size_t c = 0; c < d; ++c) {
            mixed[c] = 0;
        }
        for (size_t j = lo; j <= hi; ++j) {
            for (size_t c = 0; c < d; ++c) {
                mixed[c] += scores[j] * static_cast<float>(v[(j * d) + c]);
            }
        }
        for (size_t c = 0; c < d; ++c) {
            const float value =
                v_scale * (mixed[c] - static_cast<float>(v_zero));
            out[(row * d) + c] = quant::to_int8(value, out_scale);
        }
    }
}
