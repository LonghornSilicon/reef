// Max pool 5x5 stride 1 pad 2, three times in a row: the YOLO SPPF block on
// a 20x20xC map (C = 128 assumed for n-scale YOLO11n/YOLO26n). Each pass
// keeps the size. Vector-unit operator (no matrix engine).
//
// int4 activations, NHWC, packed per pixel with the ops.hpp row-pair
// convention (byte j = channel j low nibble, channel j + C/2 high nibble).
// Per-nibble max straight on packed bytes, as in maxpool_stem.cc:
//   max_raw (signed int8 max of raw bytes) -> correct high nibble
//   max_lo  (max of b << 4)               -> correct low nibble (<< 4)
//   out = (max_raw & 0xF0) | (max_lo >>u 4)
//
// Work unit: one 32-byte channel group (one m2 register group) of one output
// row. Separable (ALGO=0): per column, the 5-row vertical max is done in
// registers from 5 row loads; the 5-tap horizontal max slides over a window
// of 5 column maxima kept in registers. So each input byte is loaded 5x
// (once per output row that needs it) and nothing intermediate is stored.
// Direct (ALGO=1): 25 window loads and 24 maxes per output, no reuse.
// Edge rows use duplicate row pointers (max is idempotent); edge columns use
// a -inf splat.
//
//   -DRESIDENT=1  the whole 25.6 KB int4 map lives in DTCM: one load, three
//                 in-place passes. In-place needs the 2 original rows above
//                 the current one: they are copied into a 2-row ring as the
//                 vertical pass reads them (1 extra store per input byte).
//     -DSTORE_ALL=1 (default) store every pool's output (SPPF concatenates
//                 x, y1, y2, y3): 1 load + 3 stores. 0: only the last pool.
//   -DRESIDENT=0  stream each pass PSRAM -> PSRAM through a 6-row DTCM ring
//                 (rows y-2..y+2 in use, y+3 in flight: double-buffered),
//                 output rows double-buffered: 3 loads + 3 stores.
//   -DINT8=1      int8 activations (streaming only: 51 KB does not fit DTCM)
//   -DPASSES=n    number of pools (default 3)
//   -DUNROLL=n    column-loop unroll (default 20 = W)
#include "ops.hpp"

using namespace ops;

#ifndef H
#define H 20
#endif
#ifndef W
#define W 20
#endif
#ifndef C
#define C 128
#endif
#ifndef INT8
#define INT8 0
#endif
#ifndef RESIDENT
#define RESIDENT 1
#endif
#ifndef STORE_ALL
#define STORE_ALL 1
#endif
#ifndef ALGO
#define ALGO 0
#endif
#ifndef PASSES
#define PASSES 3
#endif
#ifndef UNROLL
#define UNROLL 20 // fully unroll the column loop: the 5-wide window rotates
                  // through registers with no moves (1.2x faster than rolled)
#endif

constexpr uint32_t kPB = INT8 ? C : C / 2; // bytes per pixel
constexpr uint32_t kRow = W * kPB;
constexpr uint32_t kG = 32;                // bytes per channel group (m2)
static_assert(kPB % kG == 0 && W >= 3 && H >= 3, "shape");
static_assert(!(RESIDENT && INT8), "int8 map does not fit the DTCM");

#if INT8
#define LO(x)
#else
#define LO(x) x
#endif

typedef vint8m2_t VI;
constexpr size_t kVl = kG;

// PSRAM: x, y1, y2, y3 (the SPPF concat inputs).
constexpr uint32_t buf(uint32_t i) { return kRam + i * H * kRow; }

#if RESIDENT
static int8_t map[H * kRow];
static int8_t ring[2][kRow];
#else
constexpr uint32_t kSlots = 6;
static int8_t ring[kSlots][kRow];
static int8_t obuf[2][kRow];
#endif

// Vertical 5-row max of column c (group offset off): raw and lo streams.
// COPY: also store row 0 (the current row) into cp (resident ring copy).
template <bool COPY>
E2E_INLINE void vert(const int8_t* const* rows, uint32_t o, int8_t* cp, VI& raw, VI& lo) {
    const VI r0 = __riscv_vle8_v_i8m2(rows[0] + o, kVl);
    raw = r0;
    LO(lo = __riscv_vsll_vx_i8m2(r0, 4, kVl);)
#pragma GCC unroll 4
    for (int k = 1; k < 5; ++k) {
        const VI r = __riscv_vle8_v_i8m2(rows[k] + o, kVl);
        raw = __riscv_vmax_vv_i8m2(raw, r, kVl);
        LO(lo = __riscv_vmax_vv_i8m2(lo, __riscv_vsll_vx_i8m2(r, 4, kVl), kVl);)
    }
    if (COPY) __riscv_vse8_v_i8m2(cp + o, r0, kVl);
    (void)lo;
}

E2E_INLINE void emit(int8_t* out, VI raw, VI lo) {
#if INT8
    (void)lo;
    __riscv_vse8_v_i8m2(out, raw, kVl);
#else
    const vuint8m2_t hi = __riscv_vand_vx_u8m2(__riscv_vreinterpret_v_i8m2_u8m2(raw), 0xF0, kVl);
    const vuint8m2_t l = __riscv_vsrl_vx_u8m2(__riscv_vreinterpret_v_i8m2_u8m2(lo), 4, kVl);
    __riscv_vse8_v_u8m2(reinterpret_cast<uint8_t*>(out), __riscv_vor_vv_u8m2(hi, l, kVl), kVl);
#endif
}

#define VMAX(a, b) __riscv_vmax_vv_i8m2(a, b, kVl)

// One output row. rows[0] is the current row, rows[1..4] the others
// (duplicates at the edges).
template <bool COPY>
[[maybe_unused]] static void row_sep(const int8_t* const* rows, int8_t* out, int8_t* cp) {
    const VI neg_r = __riscv_vmv_v_x_i8m2(-128, kVl);
    const VI neg_l = __riscv_vmv_v_x_i8m2(-128, kVl);
    for (uint32_t g = 0; g < kPB; g += kG) {
        VI r0 = neg_r, r1 = neg_r, r2, r3, r4, l0 = neg_l, l1 = neg_l, l2, l3, l4;
        l2 = l3 = l4 = neg_l;
        (void)l0; (void)l1;
        vert<COPY>(rows, g, cp, r2, l2);
        vert<COPY>(rows, kPB + g, cp, r3, l3);
#ifdef UNROLL
#define PRAGMA_(x) _Pragma(#x)
#define PRAGMA(x) PRAGMA_(x)
        PRAGMA(GCC unroll UNROLL)
#endif
        for (uint32_t x = 0; x < W; ++x) {
            if (x + 2 < W) {
                vert<COPY>(rows, (x + 2) * kPB + g, cp, r4, l4);
            } else {
                r4 = neg_r;
                l4 = neg_l;
            }
            const VI mr = VMAX(VMAX(VMAX(r0, r1), VMAX(r2, r3)), r4);
            VI ml = neg_l;
            LO(ml = VMAX(VMAX(VMAX(l0, l1), VMAX(l2, l3)), l4);)
            emit(out + x * kPB + g, mr, ml);
            r0 = r1; r1 = r2; r2 = r3; r3 = r4;
            LO(l0 = l1; l1 = l2; l2 = l3; l3 = l4;)
        }
    }
}

// Direct 5x5 window: 25 loads, 24 maxes per stream per output.
[[maybe_unused]] static void row_direct(const int8_t* const* rows, int8_t* out) {
    for (uint32_t g = 0; g < kPB; g += kG) {
        for (uint32_t x = 0; x < W; ++x) {
            const uint32_t x0 = x >= 2 ? x - 2 : 0, x1 = x + 2 < W ? x + 2 : W - 1;
            VI mr = __riscv_vmv_v_x_i8m2(-128, kVl), ml = mr;
            for (int k = 0; k < 5; ++k)
                for (uint32_t c = x0; c <= x1; ++c) {
                    const VI v = __riscv_vle8_v_i8m2(rows[k] + c * kPB + g, kVl);
                    mr = VMAX(mr, v);
                    LO(ml = VMAX(ml, __riscv_vsll_vx_i8m2(v, 4, kVl));)
                }
            emit(out + x * kPB + g, mr, ml);
        }
    }
}

E2E_INLINE uint32_t clampr(int32_t r) { return r < 0 ? 0 : (r >= int32_t(H) ? H - 1 : r); }

int main() {
    roi_begin();
#if RESIDENT
    for (uint32_t r = 0; r < 3; ++r) dma(buf(0) + r * kRow, addr(map + r * kRow), kRow);
    dma_wait();
    for (uint32_t p = 0; p < PASSES; ++p) {
        for (uint32_t y = 0; y < H; ++y) {
            if (p == 0 && y + 3 < H) dma(buf(0) + (y + 3) * kRow, addr(map + (y + 3) * kRow), kRow);
            // Original rows y-2, y-1 come from the ring (rows < 0 -> row y).
            const int8_t* rows[5] = {
                map + y * kRow,
                y >= 2 ? ring[y & 1] : map + y * kRow,
                y >= 1 ? ring[(y + 1) & 1] : map + y * kRow,
                map + clampr(y + 1) * kRow,
                map + clampr(y + 2) * kRow,
            };
            row_sep<true>(rows, map + y * kRow, ring[y & 1]);
            dma_wait(); // prefetch landed; previous row's store done
            if (STORE_ALL || p == PASSES - 1)
                dma(addr(map + y * kRow), buf(p + 1) + y * kRow, kRow);
        }
        mark();
    }
    dma_wait();
#else
    for (uint32_t p = 0; p < PASSES; ++p) {
        const uint32_t src = buf(p), dst = buf(p + 1);
        for (uint32_t r = 0; r < 3; ++r) dma(src + r * kRow, addr(ring[r]), kRow);
        dma_wait();
        for (uint32_t y = 0; y < H; ++y) {
            if (y + 3 < H) dma(src + (y + 3) * kRow, addr(ring[(y + 3) % kSlots]), kRow);
            const int8_t* rows[5] = {
                ring[y % kSlots],
                ring[clampr(int32_t(y) - 2) % kSlots],
                ring[clampr(int32_t(y) - 1) % kSlots],
                ring[clampr(y + 1) % kSlots],
                ring[clampr(y + 2) % kSlots],
            };
            int8_t* out = obuf[y & 1];
#if ALGO == 1
            row_direct(rows, out);
#else
            row_sep<false>(rows, out, nullptr);
#endif
            dma_wait();
            dma(addr(out), dst + y * kRow, kRow);
        }
        dma_wait();
        mark();
    }
#endif
    roi_end();
    return 0;
}
