// RVV matmul for Coral NPU, built on rvv_helpers.h.
#pragma once

#include <stddef.h>
#include <stdint.h>

// C[n x m] = A[n x k] * B[k x m] (uint8), all row-major. Any n, k, m.

// uint8 A -> uint32 C. k must stay below 66,051 to rule out overflow
// (255 * 255 * k < 2^32).
extern "C" void matmul_rvv(size_t n, size_t k, size_t m, const uint8_t* a,
                           const uint8_t* b, uint32_t* c);

// int8 A -> int32 C, the fallback for the signed score matmul (q . kt).
// k must stay below 65,794 (128 * 255 * k < 2^31).
extern "C" void matmul_rvv_s8u8(size_t n, size_t k, size_t m, const int8_t* a,
                                const uint8_t* b, int32_t* c);
