// rvv_helpers.h — readable wrappers for the RVV intrinsics the kernels use.
#pragma once
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#define RVV_INLINE static inline __attribute__((always_inline))

namespace rvv {

// Types: one place decides the element widths and LMULs used in the kernel
using vu8 = vuint8m1_t;   // a row of B, as loaded
using vu16 = vuint16m2_t; // widened row
using vu32 = vuint32m4_t; // accumulator row of C

RVV_INLINE size_t set_vl(size_t n) { return __riscv_vsetvl_e8m1(n); }
RVV_INLINE vu32 zeros(size_t vl) { return __riscv_vmv_v_x_u32m4(0, vl); }
RVV_INLINE vu16 load_widen(const uint8_t* p, size_t vl) {
    return __riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(p, vl), vl);
}
RVV_INLINE vu32 mac(vu32 acc, uint16_t a, vu16 row, size_t vl) {
    return __riscv_vwmaccu_vx_u32m4(acc, a, row, vl);
}
RVV_INLINE void store(uint32_t* p, vu32 v, size_t vl) {
    __riscv_vse32_v_u32m4(p, v, vl);
}

} // namespace rvv
