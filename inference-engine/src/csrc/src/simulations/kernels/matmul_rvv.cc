// RVV matmul for Coral NPU, built on rvv_helpers.h.
//
// C[n x m] = A[n x k] * B[k x m] (uint8), all row-major.
//
// Each pass handles one strip of vl columns (vl <= 16 with VLEN = 128) and
// computes that strip of every row of C, accumulating one row of B at a time.
// The signed variant is the same loop with a signed widening multiply-add.
#include "matmul_rvv.h"

#include "rvv_helpers.h"

extern "C" void matmul_rvv(size_t n, size_t k, size_t m, const uint8_t* a,
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

extern "C" void matmul_rvv_s8u8(size_t n, size_t k, size_t m, const int8_t* a,
                                const uint8_t* b, int32_t* c) {
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
