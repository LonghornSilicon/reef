// LayerNorm for Coral NPU, built on quant_helpers.h.
#pragma once

#include <stddef.h>
#include <stdint.h>

// out[r][c] = gamma[c] * (x[r][c] - mean_r) / sqrt(var_r + eps) + beta[c]
// for each of `rows` rows of width d, all row-major.
//
// Quantization (all zero point 0): x is int8 with x_scale, gamma and beta
// are int8 with gamma_scale and beta_scale, out is int8 with out_scale.
// eps = params::kLayernormEps (1e-5, GPT-2's).
//
// Stats: the row sum and sum of squares are exact int32 (d < 131,072 keeps
// them in range), then mean, variance and 1/sqrt are float. Future work:
// integer-only stats (integer mean and variance plus a fixed-point inverse
// square root), which would drop the FPU from this kernel.
extern "C" void layernorm(size_t rows, size_t d, const int8_t* x, float x_scale,
                          const int8_t* gamma, float gamma_scale,
                          const int8_t* beta, float beta_scale, float out_scale,
                          int8_t* out);
