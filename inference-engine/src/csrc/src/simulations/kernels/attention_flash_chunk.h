// Chunked flash attention for Coral NPU, built on quant_helpers.h.
#pragma once

#include <stddef.h>
#include <stdint.h>

// attention_flash split into steps the host can drive, so K and V stream
// through DTCM in blocks: at n_ctx = 1024 one head's K and V are 64 KB each,
// twice the 32 KB DTCM. Same quantization, masking and results (up to float
// rounding) as attention_serial; see attention_serial.h.
//
// Running state per query row, owned by the host between calls:
//   m[nq]       running max score
//   l[nq]       running softmax denominator; 0 means no key seen yet
//   acc[nq x d] running probability-weighted sum of v (zero point not
//               yet removed)
// Zero l and acc before the first block. m needs no initial value.

// One block: keys/values at absolute positions [kv_start, kv_start +
// kv_len). kt is that block of K transposed, [d x kv_len] row-major; v is
// [kv_len x d]. Each query row (absolute position query_start + row)
// folds in the keys it can see, [max(lo, kv_start), min(pos, kv_end)].
// Blocks may come in any order.
extern "C" void attention_flash_chunk(size_t nq, size_t d, size_t query_start,
                                      size_t window, size_t kv_start,
                                      size_t kv_len, const int8_t* q,
                                      const uint8_t* kt, int32_t k_zero,
                                      const uint8_t* v, float score_scale,
                                      float* m, float* l, float* acc);

// After the last block: out = v_scale * (acc / l - v_zero), requantized to
// int8 with out_scale. Every row must have seen at least one key (row i
// always sees its own position, so streaming all blocks up to it does).
extern "C" void attention_flash_finish(size_t nq, size_t d, const float* l,
                                       const float* acc, int32_t v_zero,
                                       float v_scale, float out_scale,
                                       int8_t* out);
