// Serial attention for Coral NPU: the unfused reference the flash kernel is
// checked against for accuracy and, later, cycle counts.
//
// Per query row: int8 dot products build an int32 score row, the
// standalone online_softmax kernel turns the visible slice into uint8
// probabilities (scale 1/255), and those weight the value rows in int32.
// Only the visible positions [lo, hi] are ever touched, so masked
// positions need no sentinel scores.
//
// Every stage now has the data types the VME port needs: the score stage
// maps to matmul_vme_s8u8 plus the per-row k_zero * q_sum fixup, and the
// value stage to matmul_vme_u8 (uint8 probs x uint8 v) plus a per-row
// v_zero * sum(p) fixup. That port is a sibling kernel; this scalar one
// stays the reference it is checked against.
#include "attention_serial.h"

#include "online_softmax.h"
#include "quant_helpers.h"

namespace {

// Stack bounds for one head; see the limits note in the header.
constexpr size_t kMaxHeadDim = 128;
constexpr size_t kMaxKeys = 1024;

} // namespace

extern "C" void attention_serial(size_t nq, size_t nk, size_t d,
                                 size_t query_start, size_t window,
                                 const int8_t* q, const uint8_t* kt,
                                 int32_t k_zero, const uint8_t* v,
                                 int32_t v_zero, float score_scale,
                                 float v_scale, float out_scale, int8_t* out) {
    int32_t scores[kMaxKeys];
    uint8_t probs[kMaxKeys];
    int32_t mixed[kMaxHeadDim];
    for (size_t row = 0; row < nq; ++row) {
        const size_t pos = query_start + row;
        const size_t lo = (pos + 1 > window) ? pos + 1 - window : 0;
        const size_t hi = pos; // inclusive; pos < nk per the cache invariant

        // zero-point lift: q . (k - k_zero) = q . k - k_zero * sum(q)
        int32_t q_sum = 0;
        for (size_t c = 0; c < d; ++c) {
            q_sum += q[(row * d) + c];
        }

        // int8 dot products; online_softmax applies score_scale
        for (size_t j = lo; j <= hi; ++j) {
            int32_t acc = 0;
            for (size_t c = 0; c < d; ++c) {
                acc += static_cast<int32_t>(q[(row * d) + c]) *
                       static_cast<int32_t>(kt[(c * nk) + j]);
            }
            scores[j] = acc - (k_zero * q_sum);
        }

        online_softmax(&scores[0], lo, hi, score_scale, &probs[0]);

        // probability-weighted value sum in int32. v's zero point lifts out
        // as v_zero * sum(p); the uint8 probs only sum to ~255, so use the
        // actual sum rather than assuming 255.
        int32_t p_sum = 0;
        for (size_t c = 0; c < d; ++c) {
            mixed[c] = 0;
        }
        for (size_t j = lo; j <= hi; ++j) {
            p_sum += probs[j];
            for (size_t c = 0; c < d; ++c) {
                mixed[c] += probs[j] * static_cast<int32_t>(v[(j * d) + c]);
            }
        }
        const float scale = v_scale / 255.0F; // probs carry scale 1/255
        for (size_t c = 0; c < d; ++c) {
            const float value =
                scale * static_cast<float>(mixed[c] - (v_zero * p_sum));
            out[(row * d) + c] = quant::to_int8(value, out_scale);
        }
    }
}
