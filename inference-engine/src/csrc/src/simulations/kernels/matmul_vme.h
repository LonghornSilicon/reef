// VME (Zvt) matmul for Coral NPU, built on vme_helpers.h.
#pragma once

#include <stddef.h>
#include <stdint.h>

// C[n x m] (int32) = A[n x k] (8-bit) * B[k x m] (uint8), all row-major.
// Requirements: n % 16 == 0, m % 16 == 0, k % 4 == 0 (true for GPT-2 dims).
// Other shapes are not checked: they read and write past the matrices.

// uint8 A x uint8 B (vtmmu.tvv)
extern "C" void matmul_vme_u8(size_t n, size_t k, size_t m, const uint8_t* a,
                              const uint8_t* b, int32_t* c);

// int8 A x uint8 B (vtmms.tvv). VME has no int8 x int8 matmul.
extern "C" void matmul_vme_s8u8(size_t n, size_t k, size_t m, const int8_t* a,
                                const uint8_t* b, int32_t* c);
