// The one matmul entry point for Coral NPU kernels (attention and MLP).
#pragma once

#include <stddef.h>
#include <stdint.h>

// C[n x m] (int32) = A[n x k] * B[k x m] (uint8), all row-major. A is int8
// when signed_a (pass the int8 buffer cast to uint8_t*), uint8 otherwise.
//
// Dispatch:
// - n % 16 == 0, m % 16 == 0 and k % 4 == 0: the matrix engine
//   (matmul_vme_s8u8 / matmul_vme_u8).
// - anything else, e.g. decode's n = 1: the vector unit (matmul_rvv_s8u8 /
//   matmul_rvv). We don't pad to VME's shape: padding A to 16 rows needs a
//   16 x k scratch copy plus a copy-out of C, in a 32 KB DTCM that already
//   holds A, B and C, and 15 of the 16 rows would be wasted work.
//
// Limit: k <= 33,025 on the uint8 RVV path, whose uint32 sums are returned
// as int32 (255 * 255 * k < 2^31). GPT-2's largest k is 3072.
extern "C" void matmul(size_t n, size_t k, size_t m, const uint8_t* a,
                       bool signed_a, const uint8_t* b, int32_t* c);
