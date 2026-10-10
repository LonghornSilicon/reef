// RVV matmul for Coral NPU, built on rvv_helpers.h.
#pragma once

#include <stddef.h>
#include <stdint.h>

// C[n x m] (uint32) = A[n x k] (uint8) * B[k x m] (uint8), all row-major.
// Any n, k, m. Accumulates in uint32, so k must stay below 66,051 to rule out
// overflow (255 * 255 * k < 2^32).
extern "C" void matmul_rvv(size_t n, size_t k, size_t m, const uint8_t* a,
                           const uint8_t* b, uint32_t* c);
