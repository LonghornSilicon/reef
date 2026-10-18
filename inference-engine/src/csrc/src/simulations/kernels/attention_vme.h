// Block-flash attention on the matrix engine, built on matmul.h.
#pragma once

#include <stddef.h>
#include <stdint.h>

// attention_flash_chunk with both matmuls on the matrix engine: same
// inputs, masking, running state (m, l, acc) and finish step
// (attention_flash_finish), so the host streams K/V blocks the same way.
// Results match attention_serial within one int8 step.
//
// Per KV block at absolute positions [kv_start, kv_start + kv_len):
//   1. S = q . kt  via matmul() (int8 x uint8 -> int32; the matrix engine
//      for nq % 16 == 0, the vector unit for decode's nq = 1)
//   2. per row: fix up k_zero * sum(q), apply score_scale and the causal /
//      window bounds, take the block max, and update the running max m
//   3. P = round(255 * exp(s - m)) as uint8 (0 for masked keys)
//   4. PV = P . v  via matmul() (uint8 x uint8 -> int32)
//   5. acc = acc * exp(m_old - m) + PV / 255, l = l * exp(m_old - m) +
//      sum(P) / 255
// acc holds the probability-weighted sum of raw v (zero point not yet
// removed), exactly like attention_flash_chunk, so attention_flash_finish
// applies unchanged.
//
// Both matmuls run on the matrix engine when nq % 16 == 0, kv_len % 16 == 0
// and d % 16 == 0 (GPT-2 prefill: nq = 16, 64-key blocks, d = 64); any other
// shape (decode's nq = 1) falls back to the vector unit inside matmul().
// nq <= 64. Zero l and acc before the first block.
//
// scratch: nq * max(kv_len, d) int32 (scores, then P . V) followed by
// nq * kv_len bytes (P). The host places it in DTCM next to the block.
extern "C" void attention_vme_block(size_t nq, size_t d, size_t query_start,
                                    size_t window, size_t kv_start,
                                    size_t kv_len, const int8_t* q,
                                    const uint8_t* kt, int32_t k_zero,
                                    const uint8_t* v, float score_scale,
                                    int32_t* scratch, float* m, float* l,
                                    float* acc);
