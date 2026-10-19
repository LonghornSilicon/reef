// Block-flash attention on the matrix engine: scores and P . V through
// matmul(), exp from softmax_lut's tables, the online-softmax rescale once
// per block. Header-only; built on matmul.h and pointwise.h.
//
// The host streams K/V through DTCM one block at a time (at n_ctx = 1024
// one head's K and V are 64 KB each, twice the 32 KB DTCM), calling
// vme_attention_block per block and attention_flash_finish once at the end.
//
// Running state per query row, owned by the host between calls:
//   m[nq]       running max score
//   l[nq]       running softmax denominator; 0 means no key seen yet
//   acc[nq x d] running probability-weighted sum of v (zero point not
//               yet removed)
// Zero l and acc before the first block. m needs no initial value.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "matmul.h"
#include "pointwise.h"
#include "quant_helpers.h"

namespace reef_detail {

constexpr size_t kMaxAttentionRows = 64; // nq limit

// Steps 2-3 for one query row: k_zero fixup, mask to the visible columns
// [first, last), update the running max m, and write uint8 probabilities
// (0 for masked keys). Every exp comes from exp_lut on an integer score
// gap: m is always score_scale times an integer score, so dividing it back
// recovers that integer exactly. Returns the rescale factor
// exp(m_old - m_new) for the row's old state (0 when it had none, 1 when
// nothing is visible) and adds the row's probabilities into *p_sum.
inline float row_probs(const int32_t* s_row, size_t kv_len, size_t first,
                       size_t last, int32_t lift, float score_scale,
                       const uint16_t* exp_table, float l, float* m,
                       uint8_t* p_row, int32_t* p_sum) {
    for (size_t j = 0; j < kv_len; ++j) {
        p_row[j] = 0;
    }
    *p_sum = 0;
    if (first >= last) {
        return 1.0F; // this block is masked out for this row
    }
    int32_t s_max = s_row[first];
    for (size_t j = first + 1; j < last; ++j) {
        s_max = s_row[j] > s_max ? s_row[j] : s_max;
    }
    const int32_t block_max = s_max - lift;
    float correction = 0.0F;
    int32_t m_new = block_max;
    if (l != 0.0F) {
        const float x = *m / score_scale;
        const auto m_old = static_cast<int32_t>(x < 0 ? x - 0.5F : x + 0.5F);
        m_new = block_max > m_old ? block_max : m_old;
        correction = static_cast<float>(exp_lut(
                         static_cast<uint32_t>(m_new - m_old), exp_table)) *
                     (1.0F / 32768.0F);
    }
    *m = score_scale * static_cast<float>(m_new);
    for (size_t j = first; j < last; ++j) {
        const uint32_t e = exp_lut(
            static_cast<uint32_t>(m_new - (s_row[j] - lift)), exp_table);
        p_row[j] = static_cast<uint8_t>(((255U * e) + (1U << 14U)) >> 15U);
        *p_sum += p_row[j];
    }
    return correction;
}

} // namespace reef_detail

// One K/V block at absolute positions [kv_start, kv_start + kv_len). kt is
// that block of K transposed, [d x kv_len] row-major; v is [kv_len x d].
// Query row i sits at absolute position query_start + i and sees keys in
// [pos + 1 - window, pos] (causal, optional sliding window).
//
//   1. S = q . kt  via matmul() (int8 x uint8 -> int32; the matrix engine
//      for nq % 16 == 0, the vector unit for decode's nq = 1)
//   2. per row: fix up k_zero * sum(q), apply score_scale and the causal /
//      window bounds, take the block max, and update the running max m
//   3. P = round(255 * exp(s - m)) as uint8 (0 for masked keys), with exp
//      read from exp_table on the integer score gap, so no expf per score
//   4. PV = P . v  via matmul() (uint8 x uint8 -> int32)
//   5. acc = acc * exp(m_old - m) + PV / 255, l = l * exp(m_old - m) +
//      sum(P) / 255
//
// Both matmuls run on the matrix engine when nq % 16 == 0, kv_len % 16 == 0
// and d % 16 == 0 (GPT-2 prefill: nq = 16, 64-key blocks, d = 64); any other
// shape (decode's nq = 1) falls back to the vector unit inside matmul().
// nq <= 64.
//
// exp_table: kSoftmaxLutEntries entries from softmax_lut_table(score_scale),
// built once per layer and kept in DTCM.
// scratch: nq * max(kv_len, d) int32 (scores, then P . V) followed by
// nq * kv_len bytes (P). The host places it in DTCM next to the block.
REEF_KERNEL void vme_attention_block(size_t nq, size_t d, size_t query_start,
                                     size_t window, size_t kv_start,
                                     size_t kv_len, const int8_t* q,
                                     const uint8_t* kt, int32_t k_zero,
                                     const uint8_t* v, float score_scale,
                                     const uint16_t* exp_table,
                                     int32_t* scratch, float* m, float* l,
                                     float* acc) {
    int32_t* scores = scratch; // [nq x kv_len], later P . V [nq x d]
    auto* probs = reinterpret_cast<uint8_t*>(
        scratch + (nq * (kv_len > d ? kv_len : d))); // [nq x kv_len]
    float correction[reef_detail::kMaxAttentionRows];
    int32_t p_sum[reef_detail::kMaxAttentionRows];

    // 1. S = q . kt on the matrix engine
    matmul(nq, d, kv_len, reinterpret_cast<const uint8_t*>(q), true, kt,
           scores);

    // 2-3. per row: fixup, mask, running max, uint8 probabilities
    for (size_t row = 0; row < nq; ++row) {
        const size_t pos = query_start + row;
        const size_t lo = (pos + 1 > window) ? pos + 1 - window : 0;
        // visible columns of this block: [first, last), relative
        const size_t first = lo > kv_start ? lo - kv_start : 0;
        size_t last = 0;
        if (pos + 1 > kv_start) {
            last = pos + 1 - kv_start < kv_len ? pos + 1 - kv_start : kv_len;
        }
        int32_t q_sum = 0;
        for (size_t c = 0; c < d; ++c) {
            q_sum += q[(row * d) + c];
        }
        correction[row] = reef_detail::row_probs(
            scores + (row * kv_len), kv_len, first, last, k_zero * q_sum,
            score_scale, exp_table, l[row], &m[row], probs + (row * kv_len),
            &p_sum[row]);
    }

    // 4. PV = P . v on the matrix engine (reuses the score buffer)
    int32_t* pv = scratch;
    matmul(nq, kv_len, d, probs, false, v, pv);

    // 5. fold the block into the running state; P carries scale 1/255
    constexpr float kInv255 = 1.0F / 255.0F;
    for (size_t row = 0; row < nq; ++row) {
        const float corr = correction[row];
        l[row] = (l[row] * corr) + (static_cast<float>(p_sum[row]) * kInv255);
        for (size_t c = 0; c < d; ++c) {
            float& a = acc[(row * d) + c];
            a = (a * corr) + (static_cast<float>(pv[(row * d) + c]) * kInv255);
        }
    }
}

// After the last block: out = v_scale * (acc / l - v_zero), requantized to
// int8 with out_scale. Every row must have seen at least one key (row i
// always sees its own position, so streaming all blocks up to it does).
REEF_KERNEL void attention_flash_finish(size_t nq, size_t d, const float* l,
                                        const float* acc, int32_t v_zero,
                                        float v_scale, float out_scale,
                                        int8_t* out) {
    for (size_t row = 0; row < nq; ++row) {
        for (size_t c = 0; c < d; ++c) {
            const float mean_v = acc[(row * d) + c] / l[row];
            const float value = v_scale * (mean_v - static_cast<float>(v_zero));
            out[(row * d) + c] = quant::to_int8(value, out_scale);
        }
    }
}
