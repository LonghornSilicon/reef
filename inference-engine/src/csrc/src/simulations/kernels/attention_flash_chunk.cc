// Chunked flash attention for Coral NPU: the online-softmax recurrence of
// attention_flash, with its per-row state (m, l, acc) passed in and out so
// the host can stream KV blocks through DTCM. See attention_flash_chunk.h.
#include "attention_flash_chunk.h"

#include "quant_helpers.h"

namespace {

// Dequantized score of query row q_row against column col of a kt block
// with kv_len columns; q_sum carries the k_zero lift (see attention_flash).
float score(const int8_t* q_row, size_t d, const uint8_t* kt, size_t kv_len,
            size_t col, int32_t k_zero, int32_t q_sum, float score_scale) {
    int32_t dot = 0;
    for (size_t c = 0; c < d; ++c) {
        dot += static_cast<int32_t>(q_row[c]) *
               static_cast<int32_t>(kt[(c * kv_len) + col]);
    }
    return score_scale * static_cast<float>(dot - (k_zero * q_sum));
}

// Folds one key (score s, value row v_row) into a row's running state. The
// first key a row sees (l == 0) seeds it with weight exp(0) = 1.
void fold(float s, const uint8_t* v_row, size_t d, float* m, float* l,
          float* acc_row) {
    if (*l == 0.0F) {
        *m = s;
        *l = 1.0F;
        for (size_t c = 0; c < d; ++c) {
            acc_row[c] = static_cast<float>(v_row[c]);
        }
        return;
    }
    const float m_new = s > *m ? s : *m;
    const float correction = expf(*m - m_new);
    const float p = expf(s - m_new);
    *l = (*l * correction) + p;
    for (size_t c = 0; c < d; ++c) {
        acc_row[c] =
            (acc_row[c] * correction) + (p * static_cast<float>(v_row[c]));
    }
    *m = m_new;
}

} // namespace

extern "C" void attention_flash_chunk(size_t nq, size_t d, size_t query_start,
                                      size_t window, size_t kv_start,
                                      size_t kv_len, const int8_t* q,
                                      const uint8_t* kt, int32_t k_zero,
                                      const uint8_t* v, float score_scale,
                                      float* m, float* l, float* acc) {
    const size_t kv_end = kv_start + kv_len - 1; // inclusive
    for (size_t row = 0; row < nq; ++row) {
        const size_t pos = query_start + row;
        const size_t lo = (pos + 1 > window) ? pos + 1 - window : 0;
        const size_t first = lo > kv_start ? lo : kv_start;
        const size_t last = pos < kv_end ? pos : kv_end;
        if (first > last) {
            continue; // this block is masked out for this row
        }

        const int8_t* q_row = q + (row * d);
        int32_t q_sum = 0;
        for (size_t c = 0; c < d; ++c) {
            q_sum += q_row[c];
        }
        for (size_t col = first - kv_start; col <= last - kv_start; ++col) {
            const float s =
                score(q_row, d, kt, kv_len, col, k_zero, q_sum, score_scale);
            fold(s, v + (col * d), d, &m[row], &l[row], acc + (row * d));
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
