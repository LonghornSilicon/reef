#pragma once

// Shared helpers for the operator sample kernels (coral/ops/kernels/*.cc).
// npu.hpp (copied from coral/kernels) has the DMA and Zvt wrappers.
//
// Memory map for operator kernels:
//   DTCM    0x00010000  32 KB: every buffer the kernel computes on.
//   Flash   0x20000000  read only (XSPI link): weights, LUTs, biases.
//   RAM     0x24000000  read/write (PSRAM link, psram_kv.yaml): activations.
//                       Feature maps do not fit the DTCM, so they stream in
//                       and out of this RAM by DMA.
// reef-perf moves no data and checks no results: timing only.

#include "npu.hpp"

namespace ops {
using namespace npu;

constexpr uint32_t kFlash = 0x20000000; // weights (read only)
constexpr uint32_t kRam = 0x24000000;   // activations (read/write), 8 MB
constexpr uint32_t kRamBytes = 8u << 20;

// ROI markers: reef-perf records every mcycle read (JSON mcycle_reads).
// Put roi_begin() and roi_end() around the measured region; mark() between
// phases if you want a breakdown.
E2E_INLINE void mark() { (void)mcycle(); }
E2E_INLINE void roi_begin() { mark(); }
E2E_INLINE void roi_end() { mark(); }

// Turn on the matrix state (needed before any Zvt instruction).
E2E_INLINE void matrix_on() { asm volatile("csrs mstatus, %0" ::"r"(1U << 29)); }

E2E_INLINE vint32m4_t splat(int32_t value) { return __riscv_vmv_v_x_i32m4(value, 16); }
E2E_INLINE vint32m4_t load_i8(const int8_t* ptr) {
    return __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(ptr, 16), 16);
}
E2E_INLINE vint8m1_t to_i8(vint32m4_t value) {
    value = __riscv_vmin_vx_i32m4(__riscv_vmax_vx_i32m4(value, -128, 16), 127, 16);
    return __riscv_vncvt_x_x_w_i8m1(__riscv_vncvt_x_x_w_i16m2(value, 16), 16);
}

// ---- int4 (W4A4) -----------------------------------------------------------
// Zvt multiplies int8, bf16 and fp32 only (reef-perf docs/matrix.md, "Fixed by
// the ISA"); any other type configuration retires as a no-op. So int4 is a
// STORAGE format here: weights and activations live packed, 2 per byte, in
// flash, PSRAM and DTCM (half the bytes moved and stored), and are unpacked to
// int8 on the vector unit right before the matrix engine consumes them.
//
// Packing convention for every kernel (row-pair nibble packing): a packed
// group of n bytes holds two rows of n int4 values. Byte j holds element j of
// the first row in its low nibble and element j of the second row in its high
// nibble. Signed int4, range [-8, 7]. Unpacking is then two shifts, no
// permutes:  first = (b << 4) >> 4,  second = b >> 4   (arithmetic shifts).

// Unpack n packed bytes (n % 16 == 0) into 2n int8: out[0..n) = first row,
// out[n..2n) = second row.
E2E_INLINE void unpack_i4_rows(const uint8_t* packed, int8_t* out, uint32_t n) {
    for (uint32_t i = 0; i < n;) {
        const size_t vl = __riscv_vsetvl_e8m4(n - i);
        const vint8m4_t b = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vle8_v_u8m4(packed + i, vl));
        __riscv_vse8_v_i8m4(out + i, __riscv_vsra_vx_i8m4(__riscv_vsll_vx_i8m4(b, 4, vl), 4, vl), vl);
        __riscv_vse8_v_i8m4(out + n + i, __riscv_vsra_vx_i8m4(b, 4, vl), vl);
        i += vl;
    }
}

// Clamp two int32 vectors (16 lanes) to int4 and pack them as a row pair:
// lane j -> low nibble = first[j], high nibble = second[j].
E2E_INLINE vuint8m1_t pack_i4(vint32m4_t first, vint32m4_t second) {
    first = __riscv_vmin_vx_i32m4(__riscv_vmax_vx_i32m4(first, -8, 16), 7, 16);
    second = __riscv_vmin_vx_i32m4(__riscv_vmax_vx_i32m4(second, -8, 16), 7, 16);
    const vint32m4_t packed = __riscv_vor_vv_i32m4(
        __riscv_vand_vx_i32m4(first, 0xF, 16), __riscv_vsll_vx_i32m4(second, 4, 16), 16);
    return __riscv_vreinterpret_v_i8m1_u8m1(
        __riscv_vncvt_x_x_w_i8m1(__riscv_vncvt_x_x_w_i16m2(packed, 16), 16));
}

} // namespace ops
