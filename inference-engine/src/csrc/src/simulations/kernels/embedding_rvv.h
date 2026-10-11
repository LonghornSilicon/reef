// RVV embedding kernels for Coral NPU, built on rvv_helpers.h
#pragma once

#include <stddef.h>
#include <stdint.h>

// out[length x width] = table[ids[i]] for each row i (row-major, int8)
// Mirrors inference_engine::lookup_token_embeddings with the checks removed:
// the caller just assumes ids[i] < vocabulary
extern "C" void embedding_lookup_rvv(size_t length, size_t width,
                                     const uint32_t* ids, const int8_t* table,
                                     int8_t* out);

// out[i] += position_table[position_offset + i] for each row i (int8, wraps
// on overflow)
// Mirrors inference_engine::add_position_embeddings, the caller
// guarantees position_offset + length <= max_positions
extern "C" void embedding_add_position_rvv(size_t length, size_t width,
                                           size_t position_offset,
                                           const int8_t* position_table,
                                           int8_t* out);