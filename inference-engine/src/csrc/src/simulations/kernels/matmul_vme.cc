// VME (Zvt) matmul for Coral NPU, built on vme_helpers.h.
//
// C[n x m] (int32) = A[n x k] (8-bit) * B[k x m] (uint8), all row-major.
// Requirements: n % 16 == 0, m % 16 == 0, k % 4 == 0 (true for GPT-2 dims).
//
// Each 16x16 block of C lives in matrix tile mt0 while we walk along k.
// One matmul_step = 16 x 16 x 4 = 1,024 multiply-adds.
//
// Real build: compile-checked; runs only on Coral's VME RTL model.
// -DVME_EMULATE build: runs anywhere, tests the tiling logic.
#include "matmul_vme.h"

#include "vme_helpers.h"

namespace {

template <bool kSignedA>
void matmul_tiles(size_t n, size_t k, size_t m, const uint8_t* a,
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

} // namespace

extern "C" void matmul_vme_u8(size_t n, size_t k, size_t m, const uint8_t* a,
                              const uint8_t* b, int32_t* c) {
    matmul_tiles<false>(n, k, m, a, b, c);
}

extern "C" void matmul_vme_s8u8(size_t n, size_t k, size_t m, const int8_t* a,
                                const uint8_t* b, int32_t* c) {
    matmul_tiles<true>(n, k, m, reinterpret_cast<const uint8_t*>(a), b, c);
}
