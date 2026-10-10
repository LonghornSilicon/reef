#pragma once
#include <stddef.h>
#include <stdint.h>

/** Signed W8A8 linear: weights [out,in], inputs [tokens,in], scales per
 * output channel and per token. Accumulators [tokens,out] are exact INT32.
 * Returns false for zero dimensions or a reduction width of 131072 or greater.
 * Caller supplies non-overlapping buffers of the indicated sizes. */
bool linear_w8a8(size_t tokens, size_t in, size_t out, const int8_t* x,
                 const float* x_scale, const int8_t* w, const float* w_scale,
                 const float* bias, int32_t* accumulators, float* output);

/** Symmetric [-127,127] dynamic per-token quantization, ties to even.
 * Input must be finite; an all-zero token uses scale 1. */
void quantize_tokens(size_t tokens, size_t width, const float* input,
                     int8_t* output, float* scales);

/** Two linears with tanh-GELU between them, no residual or normalization.
 * Weight layout: w1 [hidden,width], w2 [width,hidden]. Inputs are already
 * quantized; outputs remain FP32. Scratch sizes are tokens*hidden for a1,
 * hidden_fp32, hidden_q; tokens for hidden_scale; tokens*width for a2/output.
 * All inputs must be finite, scales positive, and buffers non-overlapping. */
bool mlp_w8a8(size_t tokens, size_t width, size_t hidden, const int8_t* x,
              const float* x_scale, const int8_t* w1, const float* s1,
              const float* b1, const int8_t* w2, const float* s2,
              const float* b2, int32_t* a1, float* hidden_fp32,
              int8_t* hidden_q, float* hidden_scale, int32_t* a2,
              float* output);
