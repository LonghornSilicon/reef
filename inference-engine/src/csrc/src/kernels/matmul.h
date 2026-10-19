// Matmul for Coral NPU: the matrix engine (VME) for 16 x 16-aligned shapes,
// the vector unit (RVV) for the rest, and matmul() picking between them.
// Header-only; built on vme_helpers.h and rvv_helpers.h.
//
// All variants: C[n x m] = A[n x k] * B[k x m], all row-major, B uint8.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "quant_helpers.h"
#include "rvv_helpers.h"
#include "vme_helpers.h"

namespace reef_detail {

// Each 16x16 block of C lives in matrix tile mt0 while we walk along k.
// One matmul_step = 16 x 16 x 4 = 1,024 multiply-adds.
template <bool kSignedA>
inline void vme_matmul_tiles(size_t n, size_t k, size_t m, const uint8_t* a,
                             const uint8_t* b, int32_t* c) {
    constexpr uint32_t kTile = 0;
    for (size_t row0 = 0; row0 < n; row0 += vme::kTileDim) {
        for (size_t col0 = 0; col0 < m; col0 += vme::kTileDim) {
            // 1. zero the accumulator
            vme::config_tile_moves();
            vme::zero_tile<kTile>();

            // 2. accumulate over k, 4 slices at a time
            vme::config_int8_matmul();
            for (size_t k0 = 0; k0 < k; k0 += vme::kKPerStep) {
                vme::matmul_step<kTile, kSignedA>(
                    /*a_col0=*/a + (row0 * k) + k0, /*a_stride=*/k,
                    /*b_row0=*/b + (k0 * m) + col0, /*b_stride=*/m);
            }

            // 3. write the finished 16x16 block to C
            vme::config_tile_moves();
            for (uint32_t i = 0; i < vme::kTileDim; i++) {
                vme::store_tile_row<kTile>(i, c + ((row0 + i) * m) + col0);
            }
        }
    }
}

} // namespace reef_detail

// ---- Matrix engine (Zvt) ----------------------------------------------------
// Requires n % 16 == 0, m % 16 == 0, k % 4 == 0 (true for GPT-2 dims).
// Other shapes are not checked: they read and write past the matrices.
// Real build: runs only on Coral's VME RTL model. -DVME_EMULATE: runs
// anywhere, tests the tiling logic.

// uint8 A x uint8 B -> int32 C (vtmmu.tvv)
REEF_KERNEL void vme_matmul_u8(size_t n, size_t k, size_t m, const uint8_t* a,
                               const uint8_t* b, int32_t* c) {
    reef_detail::vme_matmul_tiles<false>(n, k, m, a, b, c);
}

// int8 A x uint8 B -> int32 C (vtmms.tvv). VME has no int8 x int8 matmul.
REEF_KERNEL void vme_matmul_s8u8(size_t n, size_t k, size_t m,
                                 const int8_t* a, const uint8_t* b,
                                 int32_t* c) {
    reef_detail::vme_matmul_tiles<true>(
        n, k, m, reinterpret_cast<const uint8_t*>(a), b, c);
}

// ---- Vector unit (RVV) ------------------------------------------------------
// Any n, k, m. Each pass handles one strip of vl columns (vl <= 16 with
// VLEN = 128) and computes that strip of every row of C, accumulating one
// row of B at a time.

// uint8 A -> uint32 C. k must stay below 66,051 (255 * 255 * k < 2^32).
REEF_KERNEL void matmul_rvv(size_t n, size_t k, size_t m, const uint8_t* a,
                            const uint8_t* b, uint32_t* c) {
    size_t vl = 0;
    for (size_t col0 = 0; col0 < m; col0 += vl) {
        vl = rvv::set_vl(m - col0);
        for (size_t i = 0; i < n; i++) {
            rvv::vu32 acc = rvv::zeros(vl);
            for (size_t j = 0; j < k; j++) {
                const rvv::vu16 b_row = rvv::load_widen(b + (j * m) + col0, vl);
                acc = rvv::mac(acc, a[(i * k) + j], b_row, vl);
            }
            rvv::store(c + (i * m) + col0, acc, vl);
        }
    }
}

// int8 A -> int32 C, the fallback for the signed score matmul (q . kt).
// k must stay below 65,794 (128 * 255 * k < 2^31).
REEF_KERNEL void matmul_rvv_s8u8(size_t n, size_t k, size_t m,
                                 const int8_t* a, const uint8_t* b,
                                 int32_t* c) {
    size_t vl = 0;
    for (size_t col0 = 0; col0 < m; col0 += vl) {
        vl = rvv::set_vl(m - col0);
        for (size_t i = 0; i < n; i++) {
            rvv::vi32 acc = rvv::zeros_s(vl);
            for (size_t j = 0; j < k; j++) {
                const rvv::vi16 b_row =
                    rvv::load_widen_s(b + (j * m) + col0, vl);
                acc = rvv::mac_s(acc, a[(i * k) + j], b_row, vl);
            }
            rvv::store_s(c + (i * m) + col0, acc, vl);
        }
    }
}

// ---- Entry point ------------------------------------------------------------
// The one matmul for attention and the MLP. C (int32) = A * B; A is int8
// when signed_a (pass the int8 buffer cast to uint8_t*), uint8 otherwise.
//
// Dispatch:
// - n % 16 == 0, m % 16 == 0 and k % 4 == 0: the matrix engine.
// - anything else, e.g. decode's n = 1: the vector unit. We don't pad to
//   VME's shape: padding A to 16 rows needs a 16 x k scratch copy plus a
//   copy-out of C, in a 32 KB DTCM that already holds A, B and C, and 15 of
//   the 16 rows would be wasted work.
//
// Limit: k <= 33,025 on the uint8 RVV path, whose uint32 sums are returned
// as int32 (255 * 255 * k < 2^31). GPT-2's largest k is 3072.
REEF_KERNEL void matmul(size_t n, size_t k, size_t m, const uint8_t* a,
                        bool signed_a, const uint8_t* b, int32_t* c) {
    const auto* a_s8 = reinterpret_cast<const int8_t*>(a);
    const bool fits_vme = n % vme::kTileDim == 0 && m % vme::kTileDim == 0 &&
                          k % vme::kKPerStep == 0;
    if (fits_vme && signed_a) {
        vme_matmul_s8u8(n, k, m, a_s8, b, c);
    } else if (fits_vme) {
        vme_matmul_u8(n, k, m, a, b, c);
    } else if (signed_a) {
        matmul_rvv_s8u8(n, k, m, a_s8, b, c);
    } else {
        matmul_rvv(n, k, m, a, b, reinterpret_cast<uint32_t*>(c));
    }
}
