// helpers.h — shared kernel helpers: scalar quantization, RVV wrappers,
// and VME (matrix engine) wrappers.
#pragma once
#include <stddef.h>
#include <stdint.h>

// ---- Scalar quantization ----

#define QUANT_INLINE static inline __attribute__((always_inline))

// newlib's math, declared directly: the freestanding build has no <math.h>.
extern "C" float expf(float);
extern "C" float tanhf(float);

namespace quant {

// Round a dequantized result to the nearest int8 step and clamp to range.
QUANT_INLINE int8_t to_int8(float value, float out_scale) {
    const float scaled = value / out_scale;
    const float rounded = scaled >= 0 ? scaled + 0.5F : scaled - 0.5F;
    if (rounded >= 127.0F) {
        return 127;
    }
    if (rounded <= -128.0F) {
        return -128;
    }
    return static_cast<int8_t>(rounded);
}

// ---- Fixed-point requantization: y = round(v * f) in integers only ----
// The vector unit is integer-only (Zve32x), so every requantization runs in
// fixed point, the way Coral's own int8 kernels do it: f = m * 2^-shift
// with m in [0.5, 1) as a Q31 multiplier, a rounding Q31 multiply (vsmul),
// then a rounding, saturating narrow by shift (vnclip), both
// round-to-nearest-up. The error is under 0.5 + 2^-shift output steps.
struct Requant {
    int32_t multiplier; // m as Q31
    int32_t shift;      // right shift; negative means shift v left first
};

// f's float bits give m and the exponent exactly: m = mantissa / 2^24.
QUANT_INLINE Requant requant_params(float f) {
    uint32_t bits = 0;
    __builtin_memcpy(&bits, &f, sizeof(bits));
    const uint32_t exponent = (bits >> 23U) & 0xFFU;
    if (f <= 0.0F || exponent == 0) {
        return {0, 0}; // zero (or denormal): every output rounds to 0
    }
    auto multiplier =
        static_cast<int32_t>(((bits & 0x7FFFFFU) | 0x800000U) << 7U);
    int32_t shift = 126 - static_cast<int32_t>(exponent);
    if (shift > 31) { // vnclip shifts by at most 31
        multiplier = shift - 31 > 31 ? 0 : multiplier >> (shift - 31);
        shift = 31;
    }
    return {multiplier, shift};
}

// vsmul: round(a * b / 2^31), saturated
QUANT_INLINE int32_t q31_mul(int32_t a, int32_t b) {
    const int64_t product = (static_cast<int64_t>(a) * b) + (1LL << 30);
    const int64_t r = product >> 31;
    return r > INT32_MAX ? INT32_MAX : static_cast<int32_t>(r);
}

// vnclip to int16 then int8: round(x / 2^shift), saturated to int8
QUANT_INLINE int8_t round_shift_to_int8(int32_t x, int32_t shift) {
    const int64_t half = shift > 0 ? (1LL << (shift - 1)) : 0;
    const int64_t r = (static_cast<int64_t>(x) + half) >> shift;
    if (r > 127) {
        return 127;
    }
    if (r < -128) {
        return -128;
    }
    return static_cast<int8_t>(r);
}

// The scalar model of rvv::requant, bit for bit.
QUANT_INLINE int8_t requantize(int32_t v, Requant r) {
    if (r.shift < 0) {
        v = static_cast<int32_t>(static_cast<uint32_t>(v) << -r.shift);
    }
    return round_shift_to_int8(q31_mul(v, r.multiplier),
                               r.shift > 0 ? r.shift : 0);
}

// Fraction bits q for a sum of int8 terms a_i * f_i done as integers
// (sum a_i * round(f_i * 2^q), then round_shift_to_int8 by q): the largest
// q <= 30 with 128 * sum(f_i) * 2^q < 2^30, so the sum can't overflow.
QUANT_INLINE int32_t fraction_bits(float factor_sum) {
    int32_t q = 0;
    while (q < 30 && 128.0F * factor_sum * static_cast<float>(1U << (q + 1)) <
                         1073741824.0F) {
        ++q;
    }
    return q;
}

// round(f * 2^q) for f >= 0
QUANT_INLINE int32_t to_fixed(float f, int32_t q) {
    // f >= 0, so adding 0.5 and truncating rounds
    // NOLINTNEXTLINE(bugprone-incorrect-roundings)
    return static_cast<int32_t>((f * static_cast<float>(1U << q)) + 0.5F);
}

} // namespace quant

// ---- RVV: readable wrappers for the RVV intrinsics the kernels use ----
//
// Build with -DRVV_EMULATE to replace every wrapper with a plain C model
// (16 lanes, VLEN = 128), like -DVME_EMULATE below. That tests
// strip and indexing logic on any machine, not the intrinsics themselves.

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

// ---- Elementwise kernels (int8 rows, e8m1 / e16m2 / e32m4: 16 lanes) ----
using vi8 = vint8m1_t; // a row of int8 activations

constexpr unsigned int kRnu = 0; // vxrm: round-to-nearest-up

RVV_INLINE vi8 load_i8(const int8_t* p, size_t vl) {
    return __riscv_vle8_v_i8m1(p, vl);
}
RVV_INLINE void store_i8(int8_t* p, vi8 v, size_t vl) {
    __riscv_vse8_v_i8m1(p, v, vl);
}
RVV_INLINE void store_u8(uint8_t* p, vu8 v, size_t vl) {
    __riscv_vse8_v_u8m1(p, v, vl);
}
RVV_INLINE vu8 zeros_u8(size_t vl) { return __riscv_vmv_v_x_u8m1(0, vl); }
RVV_INLINE vi32 load_i32(const int32_t* p, size_t vl) {
    return __riscv_vle32_v_i32m4(p, vl);
}
RVV_INLINE vi32 widen(vi8 v, size_t vl) {
    return __riscv_vsext_vf4_i32m4(v, vl);
}
RVV_INLINE vi32 add(vi32 a, vi32 b, size_t vl) {
    return __riscv_vadd_vv_i32m4(a, b, vl);
}
RVV_INLINE vi32 add_x(vi32 a, int32_t x, size_t vl) {
    return __riscv_vadd_vx_i32m4(a, x, vl);
}
RVV_INLINE vi32 mul_x(vi32 a, int32_t x, size_t vl) {
    return __riscv_vmul_vx_i32m4(a, x, vl);
}
// acc + x * v
RVV_INLINE vi32 mac_x(vi32 acc, int32_t x, vi32 v, size_t vl) {
    return __riscv_vmacc_vx_i32m4(acc, x, v, vl);
}
RVV_INLINE int32_t sum(vi32 v, size_t vl) {
    return __riscv_vmv_x_s_i32m1_i32(
        __riscv_vredsum_vs_i32m4_i32m1(v, __riscv_vmv_v_x_i32m1(0, 1), vl));
}
RVV_INLINE int32_t reduce_max(vi32 v, int32_t init, size_t vl) {
    return __riscv_vmv_x_s_i32m1_i32(
        __riscv_vredmax_vs_i32m4_i32m1(v, __riscv_vmv_v_x_i32m1(init, 1), vl));
}
// quant::q31_mul per lane (vsmul)
RVV_INLINE vi32 q31_mul(vi32 v, int32_t m, size_t vl) {
    return __riscv_vsmul_vx_i32m4(v, m, kRnu, vl);
}
// quant::round_shift_to_int8 per lane (vnclip twice), 0 <= shift <= 31
RVV_INLINE vi8 round_shift_to_int8(vi32 v, int32_t shift, size_t vl) {
    const vint16m2_t half =
        __riscv_vnclip_wx_i16m2(v, static_cast<size_t>(shift), kRnu, vl);
    return __riscv_vnclip_wx_i8m1(half, 0, kRnu, vl);
}
// quant::requantize per lane
RVV_INLINE vi8 requant(vi32 v, quant::Requant r, size_t vl) {
    if (r.shift < 0) {
        v = __riscv_vsll_vx_i32m4(v, static_cast<size_t>(-r.shift), vl);
    }
    return round_shift_to_int8(q31_mul(v, r.multiplier, vl),
                               r.shift > 0 ? r.shift : 0, vl);
}
// table[x + 128] per lane: a gather with x ^ 0x80 as the byte offset
RVV_INLINE vi8 lookup_i8(const int8_t* table, vi8 x, size_t vl) {
    const vu8 index =
        __riscv_vxor_vx_u8m1(__riscv_vreinterpret_v_i8m1_u8m1(x), 0x80, vl);
    return __riscv_vluxei8_v_i8m1(table, index, vl);
}

// ---- Unsigned lanes for the softmax exp tables ----
// top - s per lane, as the unsigned score gap exp_lut takes
RVV_INLINE vu32 gap(vi32 s, int32_t top, size_t vl) {
    return __riscv_vreinterpret_v_i32m4_u32m4(
        __riscv_vrsub_vx_i32m4(s, top, vl));
}
// exp_lut (kernels.h) per lane: two gathers from the Q15 tables
RVV_INLINE vu32 exp_lut(vu32 d, const uint16_t* table, size_t vl) {
    const vbool8_t too_far = __riscv_vmsgeu_vx_u32m4_b8(d, 65536U, vl);
    d = __riscv_vminu_vx_u32m4(d, 65535U, vl);
    const vu32 coarse_at =
        __riscv_vsll_vx_u32m4(__riscv_vsrl_vx_u32m4(d, 8, vl), 1, vl);
    const vu32 fine_at = __riscv_vadd_vx_u32m4(
        __riscv_vsll_vx_u32m4(__riscv_vand_vx_u32m4(d, 255U, vl), 1, vl), 512U,
        vl);
    const vu16 coarse = __riscv_vluxei32_v_u16m2(table, coarse_at, vl);
    const vu16 fine = __riscv_vluxei32_v_u16m2(table, fine_at, vl);
    vu32 e = __riscv_vwmulu_vv_u32m4(coarse, fine, vl);
    e = __riscv_vsrl_vx_u32m4(__riscv_vadd_vx_u32m4(e, 1U << 14U, vl), 15, vl);
    return __riscv_vmerge_vxm_u32m4(e, 0, too_far, vl);
}
RVV_INLINE vu32 mul_xu(vu32 v, uint32_t x, size_t vl) {
    return __riscv_vmul_vx_u32m4(v, x, vl);
}
RVV_INLINE vu32 add_xu(vu32 v, uint32_t x, size_t vl) {
    return __riscv_vadd_vx_u32m4(v, x, vl);
}
RVV_INLINE vu32 shift_right(vu32 v, size_t shift, size_t vl) {
    return __riscv_vsrl_vx_u32m4(v, shift, vl);
}
RVV_INLINE uint32_t sum_u(vu32 v, size_t vl) {
    return __riscv_vmv_x_s_u32m1_u32(
        __riscv_vredsum_vs_u32m4_u32m1(v, __riscv_vmv_v_x_u32m1(0, 1), vl));
}
// values already in [0, 255] to uint8
RVV_INLINE vu8 narrow_u8(vu32 v, size_t vl) {
    return __riscv_vnsrl_wx_u8m1(__riscv_vnsrl_wx_u16m2(v, 0, vl), 0, vl);
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

// ---- Elementwise kernels ----
using vi8 = Lanes<int8_t>;
using vu8 = Lanes<uint8_t>;

RVV_INLINE vi8 load_i8(const int8_t* p, size_t vl) {
    vi8 v{};
    for (size_t i = 0; i < vl; i++) {
        v.e[i] = p[i];
    }
    return v;
}
RVV_INLINE void store_i8(int8_t* p, vi8 v, size_t vl) {
    for (size_t i = 0; i < vl; i++) {
        p[i] = v.e[i];
    }
}
RVV_INLINE void store_u8(uint8_t* p, vu8 v, size_t vl) {
    for (size_t i = 0; i < vl; i++) {
        p[i] = v.e[i];
    }
}
RVV_INLINE vu8 zeros_u8(size_t /*vl*/) { return {}; }
RVV_INLINE vi32 load_i32(const int32_t* p, size_t vl) {
    vi32 v{};
    for (size_t i = 0; i < vl; i++) {
        v.e[i] = p[i];
    }
    return v;
}
RVV_INLINE vi32 widen(vi8 v, size_t vl) {
    vi32 w{};
    for (size_t i = 0; i < vl; i++) {
        w.e[i] = v.e[i];
    }
    return w;
}
RVV_INLINE vi32 add(vi32 a, vi32 b, size_t vl) {
    for (size_t i = 0; i < vl; i++) {
        a.e[i] += b.e[i];
    }
    return a;
}
RVV_INLINE vi32 add_x(vi32 a, int32_t x, size_t vl) {
    for (size_t i = 0; i < vl; i++) {
        a.e[i] += x;
    }
    return a;
}
RVV_INLINE vi32 mul_x(vi32 a, int32_t x, size_t vl) {
    for (size_t i = 0; i < vl; i++) {
        a.e[i] *= x;
    }
    return a;
}
RVV_INLINE vi32 mac_x(vi32 acc, int32_t x, vi32 v, size_t vl) {
    for (size_t i = 0; i < vl; i++) {
        acc.e[i] += x * v.e[i];
    }
    return acc;
}
RVV_INLINE int32_t sum(vi32 v, size_t vl) {
    int32_t s = 0;
    for (size_t i = 0; i < vl; i++) {
        s += v.e[i];
    }
    return s;
}
RVV_INLINE int32_t reduce_max(vi32 v, int32_t init, size_t vl) {
    for (size_t i = 0; i < vl; i++) {
        init = v.e[i] > init ? v.e[i] : init;
    }
    return init;
}
RVV_INLINE vi32 q31_mul(vi32 v, int32_t m, size_t vl) {
    for (size_t i = 0; i < vl; i++) {
        v.e[i] = quant::q31_mul(v.e[i], m);
    }
    return v;
}
RVV_INLINE vi8 round_shift_to_int8(vi32 v, int32_t shift, size_t vl) {
    vi8 y{};
    for (size_t i = 0; i < vl; i++) {
        y.e[i] = quant::round_shift_to_int8(v.e[i], shift);
    }
    return y;
}
RVV_INLINE vi8 requant(vi32 v, quant::Requant r, size_t vl) {
    vi8 y{};
    for (size_t i = 0; i < vl; i++) {
        y.e[i] = quant::requantize(v.e[i], r);
    }
    return y;
}
RVV_INLINE vi8 lookup_i8(const int8_t* table, vi8 x, size_t vl) {
    vi8 y{};
    for (size_t i = 0; i < vl; i++) {
        y.e[i] = table[x.e[i] + 128];
    }
    return y;
}

// ---- Unsigned lanes for the softmax exp tables ----
RVV_INLINE vu32 gap(vi32 s, int32_t top, size_t vl) {
    vu32 d{};
    for (size_t i = 0; i < vl; i++) {
        d.e[i] = static_cast<uint32_t>(top) - static_cast<uint32_t>(s.e[i]);
    }
    return d;
}
RVV_INLINE vu32 exp_lut(vu32 d, const uint16_t* table, size_t vl) {
    for (size_t i = 0; i < vl; i++) {
        if (d.e[i] >= 65536U) {
            d.e[i] = 0;
            continue;
        }
        const uint32_t coarse = table[d.e[i] >> 8U];
        const uint32_t fine = table[256U + (d.e[i] & 255U)];
        d.e[i] = ((coarse * fine) + (1U << 14U)) >> 15U;
    }
    return d;
}
RVV_INLINE vu32 mul_xu(vu32 v, uint32_t x, size_t vl) {
    for (size_t i = 0; i < vl; i++) {
        v.e[i] *= x;
    }
    return v;
}
RVV_INLINE vu32 add_xu(vu32 v, uint32_t x, size_t vl) {
    for (size_t i = 0; i < vl; i++) {
        v.e[i] += x;
    }
    return v;
}
RVV_INLINE vu32 shift_right(vu32 v, size_t shift, size_t vl) {
    for (size_t i = 0; i < vl; i++) {
        v.e[i] >>= shift;
    }
    return v;
}
RVV_INLINE uint32_t sum_u(vu32 v, size_t vl) {
    uint32_t s = 0;
    for (size_t i = 0; i < vl; i++) {
        s += v.e[i];
    }
    return s;
}
RVV_INLINE vu8 narrow_u8(vu32 v, size_t vl) {
    vu8 y{};
    for (size_t i = 0; i < vl; i++) {
        y.e[i] = static_cast<uint8_t>(v.e[i]);
    }
    return y;
}

} // namespace rvv
#endif

// ---- VME: readable wrappers for the Coral NPU VME (Zvt) matrix engine ----
//
// The toolchain does not know VME instructions yet, so each wrapper emits the
// raw encoding. Encodings and the configuration sequence follow Coral's own
// tests (Apache 2.0, google-coral/coralnpu):
//   tests/cocotb/vme_test/vme_test_utils.h
//   tests/cocotb/vme_test/vme_matmul_test_program.cc
//
// Build with -DVME_EMULATE to replace every wrapper with a plain C model of
// what the instruction does. That lets you test tiling and indexing logic on
// any machine. It does NOT test the encodings; only Coral's RTL model can.
//
// Rules for the real (non-emulated) build:
//  * VME instructions read FIXED vector registers: the matmul reads A slices
//    from v8/v10/v12/v14 and B slices from v16/v18/v20/v22; the tile-row move
//    writes v4..v7. So loads + matmul live in ONE asm block (matmul_step) and
//    the row move + store live in ONE asm block (store_tile_row).
//  * The config wrappers overwrite vtype/vl. Don't mix them with RVV
//    intrinsics in the same function unless you re-run vsetvl afterwards.
//  * Geometry below assumes VLEN = 128 (Coral's value): 16x16 tiles.

#define VME_INLINE static inline __attribute__((always_inline))

namespace vme {

constexpr uint32_t kTileDim = 16; // C tile is 16 x 16 int32 (VLEN / 8)
constexpr uint32_t kKPerStep =
    4; // int8 matmul consumes 4 k-slices per instruction

// ---------------------------------------------------------------------------
// Encoding helpers (compile-time constants)
// ---------------------------------------------------------------------------
namespace enc {
constexpr uint32_t kOpV = 0x57;  // OP-V:  config, tile moves, vtzero
constexpr uint32_t kOpVE = 0x77; // OP-VE: matrix arithmetic

// [31:26 funct6] [25 vm=1] [24:20 vs2] [19:15 vs1/rs1] [14:12 funct3] [11:7 rd]
// [6:0 opcode]
constexpr uint32_t word(uint32_t funct6, uint32_t vs2, uint32_t rs1,
                        uint32_t funct3, uint32_t rd, uint32_t opcode) {
    return (funct6 << 26) | (1u << 25) | (vs2 << 20) | (rs1 << 15) |
           (funct3 << 12) | (rd << 7) | opcode;
}
// vtmmu.tvv (unsigned A) / vtmms.tvv (signed A): tile += A^T * B, A in v8.., B
// in v16..
constexpr uint32_t matmul_int(uint32_t tile, bool signed_a) {
    return word(0x3C, 8, 16, 0, (tile << 1) | (signed_a ? 1u : 0u), kOpVE);
}
// vtzero: tile = 0
constexpr uint32_t zero(uint32_t tile) {
    return word(0x10, 30, 0, 6, tile << 1, kOpV);
}
// vtmv.v.t v4, a0: tile slice selected by a0 -> v4..v7
constexpr uint32_t kMoveTileRowToV4 = word(0x10, 31, 10, 6, 4, kOpV);

// mtype CSR value: tm[23:10] | tk[7:5] | mtwiden[1:0]
constexpr uint32_t mtype(uint32_t tm, uint32_t tk, uint32_t mtwiden) {
    return ((tm & 0x3FFF) << 10) | ((tk & 0x7) << 5) | (mtwiden & 0x3);
}
// vtype values: vma | vta | vsew | vlmul
constexpr uint32_t kVtypeE8M1 =
    0xC0; // 8-bit elements, 1 register  (int8 operands)
constexpr uint32_t kVtypeE32M4 =
    0xD2; // 32-bit elements, 4 registers (one int32 tile row)

// Tile slice selector: tile[30:27] | pattern[26:24] (0 = row) | index[23:0]
constexpr uint32_t tile_row(uint32_t tile, uint32_t row) {
    return (tile << 27) | row;
}
} // namespace enc

#ifndef VME_EMULATE
// ===========================================================================
// Real instructions
// ===========================================================================

// msetmtype: mtype <- mtype_value, vtype <- vtype_value, vl <- 0.
VME_INLINE void set_mtype(uint32_t mtype_value, uint32_t vtype_value) {
    asm volatile(".insn r 0b1010111, 0b111, 0b1000001, x0, %0, %1"
                 :
                 : "r"(mtype_value), "r"(vtype_value));
}
// msettn: tile columns (and vl) <- min(n, hardware max). Returns the value set.
VME_INLINE uint32_t set_tn(uint32_t n) {
    uint32_t out = 0;
    asm volatile(".insn r 0b1010111, 0b111, 0b1000010, %0, %1, x0"
                 : "=r"(out)
                 : "r"(n));
    return out;
}
// msettm: tile rows <- min(n, hardware max).
VME_INLINE uint32_t set_tm(uint32_t n) {
    uint32_t out = 0;
    asm volatile(".insn r 0b1010111, 0b111, 0b1000010, %0, %1, x1"
                 : "=r"(out)
                 : "r"(n));
    return out;
}
// msettk: k-slices per matmul <- min(n, 4 for int8). Use for a k % 4 tail.
VME_INLINE uint32_t set_tk(uint32_t n) {
    uint32_t out = 0;
    asm volatile(".insn r 0b1010111, 0b111, 0b1000010, %0, %1, x2"
                 : "=r"(out)
                 : "r"(n));
    return out;
}

// Turn on matrix state (mstatus.MS). Matrix instructions trap as illegal
// (mcause 2) while it is Off. Coral's startup code enables FP and vector state
// but not matrix state; call this once if you see that trap.
VME_INLINE void enable_matrix_state() {
    asm volatile("csrs mstatus, %0" : : "r"(1u << 29));
}

// Mode for whole-tile operations on int32 data (vtzero, row moves).
VME_INLINE void config_tile_moves() {
    set_mtype(enc::mtype(kTileDim, 1, /*mtwiden=*/1), enc::kVtypeE32M4);
    set_tn(kTileDim); // msetmtype zeroed vl; this sets vl = 16
}

// Mode for int8 x int8 -> int32 matmul, 4 k-slices per instruction.
// Also leaves vl = 16 for the 16-byte operand loads.
VME_INLINE void config_int8_matmul() {
    set_mtype(enc::mtype(kTileDim, kKPerStep, /*mtwiden=*/3), enc::kVtypeE8M1);
    set_tm(kTileDim);
    set_tn(kTileDim);
}

// vtzero: tile <- 0. Call after config_tile_moves().
template <uint32_t TILE> VME_INLINE void zero_tile() {
    asm volatile(".word %0" : : "i"(enc::zero(TILE)) : "memory");
}

// One accumulate step, k-slices k0..k0+3:
//   tile[r][j] += sum_{t<4} A[r][k0+t] * B[k0+t][j]     for r, j in 0..15
// a_col0 = &A[row0][k0]  (column k0 of A; rows are a_stride bytes apart)
// b_row0 = &B[k0][col0]  (row k0 of B; rows are b_stride bytes apart)
// SIGNED_A picks vtmms (int8 A) over vtmmu (uint8 A). B is always uint8.
template <uint32_t TILE, bool SIGNED_A>
VME_INLINE void matmul_step(const uint8_t* a_col0, size_t a_stride,
                            const uint8_t* b_row0, size_t b_stride) {
    asm volatile(
        "vlse8.v v8,  (%[a0]), %[as]\n" // A column k0+0 (16 rows, strided)
        "vlse8.v v10, (%[a1]), %[as]\n" // A column k0+1
        "vlse8.v v12, (%[a2]), %[as]\n" // A column k0+2
        "vlse8.v v14, (%[a3]), %[as]\n" // A column k0+3
        "vle8.v  v16, (%[b0])\n"        // B row k0+0 (16 contiguous bytes)
        "vle8.v  v18, (%[b1])\n"        // B row k0+1
        "vle8.v  v20, (%[b2])\n"        // B row k0+2
        "vle8.v  v22, (%[b3])\n"        // B row k0+3
        ".word %[mm]\n"                 // vtmmu/vtmms tile, v8, v16
        :
        : [a0] "r"(a_col0), [a1] "r"(a_col0 + 1), [a2] "r"(a_col0 + 2),
          [a3] "r"(a_col0 + 3), [as] "r"(a_stride), [b0] "r"(b_row0),
          [b1] "r"(b_row0 + b_stride), [b2] "r"(b_row0 + (2 * b_stride)),
          [b3] "r"(b_row0 + (3 * b_stride)),
          [mm] "i"(enc::matmul_int(TILE, SIGNED_A))
        : "v8", "v10", "v12", "v14", "v16", "v18", "v20", "v22", "memory");
}

// Copy tile row `row` (16 int32s) to memory. Call after config_tile_moves().
template <uint32_t TILE>
// NOLINTNEXTLINE(readability-non-const-parameter): the asm stores through dst
VME_INLINE void store_tile_row(uint32_t row, int32_t* dst) {
    register uint32_t sel asm("a0") =
        enc::tile_row(TILE, row);         // encoding reads a0
    asm volatile(".word %[mv]\n"          // vtmv.v.t v4, a0
                 "vse32.v v4, (%[dst])\n" // store v4..v7
                 :
                 : [mv] "i"(enc::kMoveTileRowToV4), "r"(sel), [dst] "r"(dst)
                 : "v4", "v5", "v6", "v7", "memory");
}

#else
// ===========================================================================
// Emulation: plain C model of each instruction, for testing kernel logic.
// ===========================================================================
// Matrix tile state. Mutable and global because it models hardware registers.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
inline int32_t g_tiles[16][kTileDim][kTileDim];

VME_INLINE void enable_matrix_state() {}
VME_INLINE void config_tile_moves() {}
VME_INLINE void config_int8_matmul() {}
VME_INLINE uint32_t set_tk(uint32_t n) { return n < kKPerStep ? n : kKPerStep; }

template <uint32_t TILE> VME_INLINE void zero_tile() {
    for (auto& row : g_tiles[TILE]) {
        for (auto& x : row) {
            x = 0;
        }
    }
}

template <uint32_t TILE, bool SIGNED_A>
VME_INLINE void matmul_step(const uint8_t* a_col0, size_t a_stride,
                            const uint8_t* b_row0, size_t b_stride) {
    for (uint32_t t = 0; t < kKPerStep; t++) {
        for (uint32_t r = 0; r < kTileDim; r++) {
            const uint8_t raw = a_col0[t + (r * a_stride)];
            const int32_t a =
                SIGNED_A ? static_cast<int32_t>(static_cast<int8_t>(raw))
                         : static_cast<int32_t>(raw);
            for (uint32_t j = 0; j < kTileDim; j++) {
                g_tiles[TILE][r][j] +=
                    a * static_cast<int32_t>(b_row0[(t * b_stride) + j]);
            }
        }
    }
}

template <uint32_t TILE>
VME_INLINE void store_tile_row(uint32_t row, int32_t* dst) {
    for (uint32_t j = 0; j < kTileDim; j++) {
        dst[j] = g_tiles[TILE][row][j];
    }
}
#endif

} // namespace vme
