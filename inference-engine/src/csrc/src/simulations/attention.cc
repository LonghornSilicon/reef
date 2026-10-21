// Attention for Coral NPU: block-flash attention on the matrix engine
// (scores and P . V through matmul(), exp from the softmax_lut_table
// tables, the online-softmax rescale once per block) and its finish step:
// the attention operator (operator/attention.hpp). See kernels.h.
#include "helpers.h"
#include "kernels.h"

namespace {

constexpr size_t kMaxRows = 64; // nq limit; see the header

// Steps 2-3 for one query row: k_zero fixup, mask to the visible columns
// [first, last), update the running max m, and write uint8 probabilities
// (0 for masked keys). Every exp comes from exp_lut on an integer score
// gap: m is always score_scale times an integer score, so dividing it back
// recovers that integer exactly. Returns the rescale factor
// exp(m_old - m_new) for the row's old state (0 when it had none, 1 when
// nothing is visible) and adds the row's probabilities into *p_sum.
float row_probs(const int32_t* s_row, size_t kv_len, size_t first, size_t last,
                int32_t lift, float score_scale, const uint16_t* exp_table,
                float l, float* m, uint8_t* p_row, int32_t* p_sum) {
    size_t vl = 0;
    for (size_t j = 0; j < kv_len; j += vl) {
        vl = rvv::set_vl(kv_len - j);
        rvv::store_u8(p_row + j, rvv::zeros_u8(vl), vl);
    }
    *p_sum = 0;
    if (first >= last) {
        return 1.0F; // this block is masked out for this row
    }
    int32_t s_max = s_row[first];
    for (size_t j = first; j < last; j += vl) {
        vl = rvv::set_vl(last - j);
        s_max = rvv::reduce_max(rvv::load_i32(s_row + j, vl), s_max, vl);
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
    // p = round(255 * exp_lut(m_new - (s - lift))), 16 keys at a time
    uint32_t total = 0;
    for (size_t j = first; j < last; j += vl) {
        vl = rvv::set_vl(last - j);
        const rvv::vu32 e = rvv::exp_lut(
            rvv::gap(rvv::load_i32(s_row + j, vl), m_new + lift, vl), exp_table,
            vl);
        const rvv::vu32 p = rvv::shift_right(
            rvv::add_xu(rvv::mul_xu(e, 255U, vl), 1U << 14U, vl), 15, vl);
        total += rvv::sum_u(p, vl);
        rvv::store_u8(p_row + j, rvv::narrow_u8(p, vl), vl);
    }
    *p_sum = static_cast<int32_t>(total);
    return correction;
}

} // namespace

extern "C" void vme_attention_block(size_t nq, size_t d, size_t query_start,
                                    size_t window, size_t kv_start,
                                    size_t kv_len, const int8_t* q,
                                    const uint8_t* kt, int32_t k_zero,
                                    const uint8_t* v, float score_scale,
                                    const uint16_t* exp_table, int32_t* scratch,
                                    float* m, float* l, float* acc) {
    int32_t* scores = scratch; // [nq x kv_len], later P . V [nq x d]
    auto* probs = reinterpret_cast<uint8_t*>(
        scratch + (nq * (kv_len > d ? kv_len : d))); // [nq x kv_len]
    float correction[kMaxRows];
    int32_t p_sum[kMaxRows];

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
        size_t vl = 0;
        for (size_t c = 0; c < d; c += vl) {
            vl = rvv::set_vl(d - c);
            q_sum += rvv::sum(
                rvv::widen(rvv::load_i8(q + (row * d) + c, vl), vl), vl);
        }
        correction[row] =
            row_probs(scores + (row * kv_len), kv_len, first, last,
                      k_zero * q_sum, score_scale, exp_table, l[row], &m[row],
                      probs + (row * kv_len), &p_sum[row]);
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

extern "C" void attention_flash_finish(size_t nq, size_t d, const float* l,
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
