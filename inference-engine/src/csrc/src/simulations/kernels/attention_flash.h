// Flash attention for Coral NPU, built on quant_helpers.h.
#pragma once

#include <stddef.h>
#include <stdint.h>

// Same contract as attention_serial (see attention_serial.h): identical
// inputs, quantization and masking, and matching results up to float
// rounding. The difference is the algorithm: one fused pass per query row
// keeps a running maximum, denominator and weighted value sum (an online
// softmax), so no score row is ever materialized and the per-row state is
// O(d) instead of O(nk).
extern "C" void attention_flash(size_t nq, size_t nk, size_t d,
                                size_t query_start, size_t window,
                                const int8_t* q, const uint8_t* kt,
                                int32_t k_zero, const uint8_t* v,
                                int32_t v_zero, float score_scale,
                                float v_scale, float out_scale, int8_t* out);
