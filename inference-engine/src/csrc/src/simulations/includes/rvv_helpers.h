// rvv_helpers.h — readable wrappers for the RVV intrinsics the kernels use.
//
// Build with -DRVV_EMULATE to replace every wrapper with a plain C model
// (16 lanes, VLEN = 128), like -DVME_EMULATE in vme_helpers.h. That tests
// strip and indexing logic on any machine, not the intrinsics themselves.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define RVV_INLINE static inline __attribute__((always_inline))

#ifndef RVV_EMULATE
#include <riscv_vector.h>

namespace rvv {

// Types: one place decides the element widths and LMULs used in the kernel
using vu8 = vuint8m1_t;   // a row of B, as loaded
using vu16 = vuint16m2_t; // widened row
using vu32 = vuint32m4_t; // accumulator row of C
using vi16 = vint16m2_t;  // widened row, for a signed A
using vi32 = vint32m4_t;  // signed accumulator row of C

RVV_INLINE size_t set_vl(size_t n) { return __riscv_vsetvl_e8m1(n); }
RVV_INLINE vu32 zeros(size_t vl) { return __riscv_vmv_v_x_u32m4(0, vl); }
RVV_INLINE vi32 zeros_s(size_t vl) { return __riscv_vmv_v_x_i32m4(0, vl); }
RVV_INLINE vu16 load_widen(const uint8_t* p, size_t vl) {
    return __riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(p, vl), vl);
}
// uint8 fits in int16, so the zero-extended row reads as signed unchanged
RVV_INLINE vi16 load_widen_s(const uint8_t* p, size_t vl) {
    return __riscv_vreinterpret_v_u16m2_i16m2(load_widen(p, vl));
}
RVV_INLINE vu32 mac(vu32 acc, uint16_t a, vu16 row, size_t vl) {
    return __riscv_vwmaccu_vx_u32m4(acc, a, row, vl);
}
RVV_INLINE vi32 mac_s(vi32 acc, int16_t a, vi16 row, size_t vl) {
    return __riscv_vwmacc_vx_i32m4(acc, a, row, vl);
}
RVV_INLINE void store(uint32_t* p, vu32 v, size_t vl) {
    __riscv_vse32_v_u32m4(p, v, vl);
}
RVV_INLINE void store_s(int32_t* p, vi32 v, size_t vl) {
    __riscv_vse32_v_i32m4(p, v, vl);
}

} // namespace rvv

#else
// ===========================================================================
// Emulation: plain C model of each wrapper, for testing kernel logic.
// ===========================================================================
namespace rvv {

constexpr size_t kLanes = 16; // e8m1 with VLEN = 128

template <typename T> struct Lanes {
    T e[kLanes];
};
using vu16 = Lanes<uint16_t>;
using vu32 = Lanes<uint32_t>;
using vi16 = Lanes<int16_t>;
using vi32 = Lanes<int32_t>;

RVV_INLINE size_t set_vl(size_t n) { return n < kLanes ? n : kLanes; }
RVV_INLINE vu32 zeros(size_t /*vl*/) { return {}; }
RVV_INLINE vi32 zeros_s(size_t /*vl*/) { return {}; }
RVV_INLINE vu16 load_widen(const uint8_t* p, size_t vl) {
    vu16 v{};
    for (size_t i = 0; i < vl; i++) {
        v.e[i] = p[i];
    }
    return v;
}
RVV_INLINE vi16 load_widen_s(const uint8_t* p, size_t vl) {
    vi16 v{};
    for (size_t i = 0; i < vl; i++) {
        v.e[i] = p[i];
    }
    return v;
}
RVV_INLINE vu32 mac(vu32 acc, uint16_t a, vu16 row, size_t vl) {
    for (size_t i = 0; i < vl; i++) {
        acc.e[i] += static_cast<uint32_t>(a) * row.e[i];
    }
    return acc;
}
RVV_INLINE vi32 mac_s(vi32 acc, int16_t a, vi16 row, size_t vl) {
    for (size_t i = 0; i < vl; i++) {
        acc.e[i] += static_cast<int32_t>(a) * row.e[i];
    }
    return acc;
}
RVV_INLINE void store(uint32_t* p, vu32 v, size_t vl) {
    for (size_t i = 0; i < vl; i++) {
        p[i] = v.e[i];
    }
}
RVV_INLINE void store_s(int32_t* p, vi32 v, size_t vl) {
    for (size_t i = 0; i < vl; i++) {
        p[i] = v.e[i];
    }
}

} // namespace rvv
#endif
