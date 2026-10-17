// GeLU for Coral NPU as a 256-entry lookup table, built on quant_helpers.h.
#pragma once

#include <stddef.h>
#include <stdint.h>

// Fills table[256] so that table[x + 128] = GeLU(x * in_scale) requantized
// to int8 with out_scale (both zero point 0). GeLU is GPT-2's tanh form:
// 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). Build it once per pair
// of scales (params::kGeluInScale, params::kGeluOutScale).
extern "C" void gelu_table(float in_scale, float out_scale, int8_t* table);

// y[i] = table[x[i] + 128] for i < n. A pure lookup, so it is exact for
// this quantization: every int8 input has its own precomputed output.
extern "C" void gelu(size_t n, const int8_t* x, const int8_t* table, int8_t* y);
