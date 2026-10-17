// Residual add for Coral NPU, built on quant_helpers.h.
#pragma once

#include <stddef.h>
#include <stdint.h>

// out[i] = a[i] * a_scale + b[i] * b_scale, requantized to int8 with
// out_scale, for i < n. All int8 with zero point 0. This is the skip
// connection around attention and the MLP, and lets a block's two branches
// carry different scales.
extern "C" void residual_add(size_t n, const int8_t* a, float a_scale,
                             const int8_t* b, float b_scale, float out_scale,
                             int8_t* out);
