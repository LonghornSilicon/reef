// Serial (unfused) attention for Coral NPU, built on quant_helpers.h.
#pragma once

#include <stddef.h>
#include <stdint.h>

// out[nq x d] (int8) = softmax(q[nq x d] . kt[d x nk]) * v[nk x d], all
// row-major, one attention head.
//
// Quantization: q is int8 with zero point 0. kt (K transposed, so it can be
// the uint8 B operand of the VME matmul) and v are uint8 with zero points
// k_zero and v_zero. score_scale turns a q.k accumulator into a real score:
// q_scale * k_scale / sqrt(d) for GPT-2, which folds its 1/sqrt(d_head) in
// here (params::kScoreScale). v_scale dequantizes v and out_scale
// requantizes the result into int8.
//
// Masking is causal, done by loop bounds instead of mask values: query row i
// sees key positions [lo, query_start + i], where lo = max(0, query_start +
// i + 1 - window). GPT-2 attends to the full causal prefix, so pass
// window >= nk (params::kWindow = n_ctx). A smaller window gives a sliding
// local window; window must be at least 1.
//
// Limits (not checked, like the matmul kernels): d <= 128, nk <= 1024, and
// query_start + nq <= nk, which the decode cache layout guarantees.
extern "C" void attention_serial(size_t nq, size_t nk, size_t d,
                                 size_t query_start, size_t window,
                                 const int8_t* q, const uint8_t* kt,
                                 int32_t k_zero, const uint8_t* v,
                                 int32_t v_zero, float score_scale,
                                 float v_scale, float out_scale, int8_t* out);
