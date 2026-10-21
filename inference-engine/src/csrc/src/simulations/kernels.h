// Int8 transformer kernels for the Coral NPU, built on helpers.h.
//
// matmul.cc     vme_matmul_u8/s8u8, matmul_rvv/_s8u8, matmul (dispatcher)
// attention.cc  vme_attention_block, attention_flash_finish, softmax_lut_table
// pointwise.cc  gelu_table/gelu, layernorm, residual_add
#pragma once

#include <stddef.h>
#include <stdint.h>

// ===========================================================================
// Matmul: C[n x m] (int32) = A[n x k] * B[k x m] (uint8), all row-major
// ===========================================================================

// Matrix engine (Zvt). Requirements: n % 16 == 0, m % 16 == 0, k % 4 == 0
// (true for GPT-2 dims). Other shapes are not checked: they read and write
// past the matrices.
// uint8 A x uint8 B (vtmmu.tvv)
extern "C" void vme_matmul_u8(size_t n, size_t k, size_t m, const uint8_t* a,
                              const uint8_t* b, int32_t* c);
// int8 A x uint8 B (vtmms.tvv). VME has no int8 x int8 matmul.
extern "C" void vme_matmul_s8u8(size_t n, size_t k, size_t m, const int8_t* a,
                                const uint8_t* b, int32_t* c);

// Vector unit. Any n, k, m.
// uint8 A -> uint32 C. k must stay below 66,051 to rule out overflow
// (255 * 255 * k < 2^32).
extern "C" void matmul_rvv(size_t n, size_t k, size_t m, const uint8_t* a,
                           const uint8_t* b, uint32_t* c);
// int8 A -> int32 C, the fallback for the signed score matmul (q . kt).
// k must stay below 65,794 (128 * 255 * k < 2^31).
extern "C" void matmul_rvv_s8u8(size_t n, size_t k, size_t m, const int8_t* a,
                                const uint8_t* b, int32_t* c);

// The one matmul entry point. A is int8 when signed_a (pass the int8 buffer
// cast to uint8_t*), uint8 otherwise.
//
// Dispatch:
// - n % 16 == 0, m % 16 == 0 and k % 4 == 0: the matrix engine
//   (vme_matmul_s8u8 / vme_matmul_u8).
// - anything else, e.g. decode's n = 1: the vector unit (matmul_rvv_s8u8 /
//   matmul_rvv). We don't pad to VME's shape: padding A to 16 rows needs a
//   16 x k scratch copy plus a copy-out of C, in a 32 KB DTCM that already
//   holds A, B and C, and 15 of the 16 rows would be wasted work.
//
// Limit: k <= 33,025 on the uint8 RVV path, whose uint32 sums are returned
// as int32 (255 * 255 * k < 2^31). GPT-2's largest k is 3072.
extern "C" void matmul(size_t n, size_t k, size_t m, const uint8_t* a,
                       bool signed_a, const uint8_t* b, int32_t* c);

// ===========================================================================
// Attention
// ===========================================================================

// Entries softmax_lut_table writes: 256 coarse, then 256 fine.
constexpr size_t kSoftmaxLutEntries = 512;

// Fills table[512] with exp(-score_scale * d) in Q15 (1.0 = 32768) for a
// score gap d = max - s split into its high and low bytes:
//   table[i]       = exp(-score_scale * 256 * i)
//   table[256 + j] = exp(-score_scale * j)
// so exp(-score_scale * d) = table[d >> 8] * table[256 + (d & 255)] is
// exact up to the Q15 rounding. Gaps of 65536 or more read as 0, which
// needs score_scale * 65536 >= 13 (exp(-13) is under 1/255 / 1000);
// GPT-2's score scale gives 20.5. Build it once per score_scale, i.e. once
// per layer, like gelu_table.
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

// Block-flash attention for one head, with both matmuls on the matrix
// engine. The host streams K and V through DTCM in blocks: at n_ctx = 1024
// one head's K and V are 64 KB each, twice the 32 KB DTCM.
//
// Quantization: q is int8 with zero point 0. kt (K transposed, so it can be
// the uint8 B operand of the matmul) and v are uint8 with zero points k_zero
// and v_zero. score_scale turns a q.k accumulator into a real score:
// q_scale * k_scale / sqrt(d) for GPT-2.
//
// Masking is causal, done by loop bounds instead of mask values: query row i
// (absolute position query_start + i) sees key positions [lo, query_start +
// i], where lo = max(0, query_start + i + 1 - window). GPT-2 attends to the
// full causal prefix (window >= n_ctx); a smaller window gives a sliding
// local window (GPT-Neo's local layers). window must be at least 1.
//
// Running state per query row, owned by the host between calls:
//   m[nq]       running max score
//   l[nq]       running softmax denominator; 0 means no key seen yet
//   acc[nq x d] running probability-weighted sum of v (zero point not
//               yet removed)
// Zero l and acc before the first block. m needs no initial value.
//
// Per KV block at absolute positions [kv_start, kv_start + kv_len), with kt
// that block of K transposed ([d x kv_len]) and v [kv_len x d]:
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
extern "C" void vme_attention_block(size_t nq, size_t d, size_t query_start,
                                    size_t window, size_t kv_start,
                                    size_t kv_len, const int8_t* q,
                                    const uint8_t* kt, int32_t k_zero,
                                    const uint8_t* v, float score_scale,
                                    const uint16_t* exp_table, int32_t* scratch,
                                    float* m, float* l, float* acc);

// After the last block: out = v_scale * (acc / l - v_zero), requantized to
// int8 with out_scale. Every row must have seen at least one key (row i
// always sees its own position, so streaming all blocks up to it does).
extern "C" void attention_flash_finish(size_t nq, size_t d, const float* l,
                                       const float* acc, int32_t v_zero,
                                       float v_scale, float out_scale,
                                       int8_t* out);

// ===========================================================================
// Pointwise
// ===========================================================================

// Fills table[256] so that table[x + 128] = GeLU(x * in_scale) requantized
// to int8 with out_scale (both zero point 0). GeLU is GPT-2's tanh form:
// 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). Build it once per pair
// of scales.
extern "C" void gelu_table(float in_scale, float out_scale, int8_t* table);

// y[i] = table[x[i] + 128] for i < n. A pure lookup, so it is exact for
// this quantization: every int8 input has its own precomputed output.
extern "C" void gelu(size_t n, const int8_t* x, const int8_t* table, int8_t* y);

// out[r][c] = gamma[c] * (x[r][c] - mean_r) / sqrt(var_r + eps) + beta[c]
// for each of `rows` rows of width d, all row-major.
//
// Quantization (all zero point 0): x is int8 with x_scale, gamma and beta
// are int8 with gamma_scale and beta_scale, out is int8 with out_scale.
// eps = 1e-5 (GPT-2's).
//
// Stats: the row sum and sum of squares are exact int32 (d < 131,072 keeps
// them in range), then mean, variance and 1/sqrt are float. Future work:
// integer-only stats (integer mean and variance plus a fixed-point inverse
// square root), which would drop the FPU from this kernel.
extern "C" void layernorm(size_t rows, size_t d, const int8_t* x, float x_scale,
                          const int8_t* gamma, float gamma_scale,
                          const int8_t* beta, float beta_scale, float out_scale,
                          int8_t* out);

// out[i] = a[i] * a_scale + b[i] * b_scale, requantized to int8 with
// out_scale, for i < n. All int8 with zero point 0. This is the skip
// connection around attention and the MLP, and lets a block's two branches
// carry different scales.
extern "C" void residual_add(size_t n, const int8_t* a, float a_scale,
                             const int8_t* b, float b_scale, float out_scale,
                             int8_t* out);
