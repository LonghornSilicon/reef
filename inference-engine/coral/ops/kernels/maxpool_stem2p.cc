// Max pool 3x3 s2 p1, TWO-PASS variant (vertical rows -> DTCM intermediates -> horizontal). Kept for comparison; see maxpool_stem.cc (fused, faster).
// Vector-unit operator (no matrix engine). Activations int4, NHWC, packed in
// PSRAM with the row-pair convention of ops.hpp applied per pixel: byte j of
// a pixel holds channel j (low nibble) and channel j + C/2 (high nibble).
//
// Per-nibble max straight on packed bytes (fused unpack/pack, no int8 copy):
//   high nibble: a signed int8 max of the RAW bytes orders by the high nibble
//                first, so max(raw)'s high nibble is the max of high nibbles.
//   low nibble : (b << 4) puts the low nibble in the high position (low bits
//                0), so max(b << 4) >> 4 is the max of low nibbles.
//   repack     : out = (max_raw & 0xF0) | (max_lo >>u 4).
// So the "unpack" is one shift per input byte and the "pack" is 3 ops per
// output byte. Padding: pad bytes are 0x88 (both nibbles -8 = -inf).
//
// Dataflow (separable, vertical then horizontal): input rows stream from
// PSRAM into a 5-slot DTCM ring (rows 2y-1, 2y, 2y+1 in use, rows 2y+2, 2y+3
// in flight: double-buffered). The vertical 3-row max runs on whole
// contiguous rows (m4, 64 B per op); the horizontal stride-2 3-tap max runs
// per output pixel. Output rows are double-buffered and DMA'd back.
//   -DINT8=1   int8 activations (no packing) for the int4-vs-int8 comparison
//   -DOWT=n    output columns per tile (56 = full row; int8 needs 28 to fit)
#include "ops.hpp"

using namespace ops;

#ifndef INT8
#define INT8 0
#endif
#ifndef H
#define H 112
#endif
#ifndef W
#define W 112
#endif
#ifndef C
#define C 64
#endif
#ifndef OWT
#define OWT (INT8 ? 28 : 56)
#endif

constexpr uint32_t kOH = H / 2, kOW = W / 2;
constexpr uint32_t kPB = INT8 ? C : C / 2;      // bytes per pixel
constexpr uint32_t kSW = 2 * OWT + 1;           // slot width: left halo + 2*OWT
constexpr uint32_t kSlot = kSW * kPB;
constexpr uint32_t kSlots = 5;
constexpr uint8_t kPad = INT8 ? 0x80 : 0x88;
static_assert(kOW % OWT == 0, "column tiles");

static uint8_t ring[kSlots][kSlot];
static int8_t vraw[kSlot];
#if !INT8
static int8_t vlo[kSlot];
#endif
static uint8_t obuf[2][OWT * kPB];

constexpr uint32_t kIn = kRam;                          // input map
constexpr uint32_t kOut = kRam + H * W * kPB;           // output map

// Load input row r, columns of tile x0, into its ring slot.
E2E_INLINE void load_row(uint32_t r, uint32_t x0) {
    uint8_t* slot = ring[r % kSlots];
    if (x0 == 0) { // left pad pixel stays 0x88; columns 0..2*OWT-1
        dma(kIn + r * W * kPB, addr(slot + kPB), 2 * OWT * kPB);
    } else {
        const uint32_t c0 = 2 * x0 - 1;
        const uint32_t n = (c0 + kSW <= W ? kSW : W - c0);
        dma(kIn + (r * W + c0) * kPB, addr(slot), n * kPB);
    }
}

// Vertical 3-row max over a whole slot (contiguous bytes).
E2E_INLINE void vertical(const uint8_t* a, const uint8_t* b, const uint8_t* c) {
    for (uint32_t i = 0; i < kSlot;) {
        const size_t vl = __riscv_vsetvl_e8m4(kSlot - i);
        const vint8m4_t ra = __riscv_vle8_v_i8m4(reinterpret_cast<const int8_t*>(a + i), vl);
        const vint8m4_t rb = __riscv_vle8_v_i8m4(reinterpret_cast<const int8_t*>(b + i), vl);
        const vint8m4_t rc = __riscv_vle8_v_i8m4(reinterpret_cast<const int8_t*>(c + i), vl);
        __riscv_vse8_v_i8m4(vraw + i, __riscv_vmax_vv_i8m4(__riscv_vmax_vv_i8m4(ra, rb, vl), rc, vl), vl);
#if !INT8
        const vint8m4_t la = __riscv_vsll_vx_i8m4(ra, 4, vl);
        const vint8m4_t lb = __riscv_vsll_vx_i8m4(rb, 4, vl);
        const vint8m4_t lc = __riscv_vsll_vx_i8m4(rc, 4, vl);
        __riscv_vse8_v_i8m4(vlo + i, __riscv_vmax_vv_i8m4(__riscv_vmax_vv_i8m4(la, lb, vl), lc, vl), vl);
#endif
        i += vl;
    }
}

// Horizontal stride-2 3-tap max: out pixel x = max(slot pixels 2x, 2x+1, 2x+2)
// (slot pixel 0 is input column 2*x0-1). The right tap is carried as the next
// left tap.
E2E_INLINE void horizontal(uint8_t* out) {
    const size_t vl = __riscv_vsetvl_e8m4(kPB);
    vint8m4_t r0 = __riscv_vle8_v_i8m4(vraw, vl);
#if !INT8
    vint8m4_t l0 = __riscv_vle8_v_i8m4(vlo, vl);
#endif
    for (uint32_t x = 0; x < OWT; ++x) {
        const uint32_t p = (2 * x + 1) * kPB;
        const vint8m4_t r1 = __riscv_vle8_v_i8m4(vraw + p, vl);
        const vint8m4_t r2 = __riscv_vle8_v_i8m4(vraw + p + kPB, vl);
        const vint8m4_t mr = __riscv_vmax_vv_i8m4(__riscv_vmax_vv_i8m4(r0, r1, vl), r2, vl);
#if INT8
        __riscv_vse8_v_i8m4(reinterpret_cast<int8_t*>(out + x * kPB), mr, vl);
#else
        const vint8m4_t l1 = __riscv_vle8_v_i8m4(vlo + p, vl);
        const vint8m4_t l2 = __riscv_vle8_v_i8m4(vlo + p + kPB, vl);
        const vint8m4_t ml = __riscv_vmax_vv_i8m4(__riscv_vmax_vv_i8m4(l0, l1, vl), l2, vl);
        const vuint8m4_t hi = __riscv_vand_vx_u8m4(__riscv_vreinterpret_v_i8m4_u8m4(mr), 0xF0, vl);
        const vuint8m4_t lo = __riscv_vsrl_vx_u8m4(__riscv_vreinterpret_v_i8m4_u8m4(ml), 4, vl);
        __riscv_vse8_v_u8m4(out + x * kPB, __riscv_vor_vv_u8m4(hi, lo, vl), vl);
        l0 = l2;
#endif
        r0 = r2;
    }
}

int main() {
    roi_begin();
    for (uint32_t x0 = 0; x0 < kOW; x0 += OWT) {
        if (x0 == 0)
            for (uint32_t s = 0; s < kSlots; ++s)
                for (uint32_t j = 0; j < kPB; ++j) ring[s][j] = kPad;
        load_row(0, x0);
        load_row(1, x0);
        dma_wait();
        for (uint32_t y = 0; y < kOH; ++y) {
            if (2 * y + 2 < H) load_row(2 * y + 2, x0); // prefetch next pair
            if (2 * y + 3 < H) load_row(2 * y + 3, x0);
            const uint32_t top = (y == 0 ? 0 : 2 * y - 1);
            vertical(ring[top % kSlots], ring[(2 * y) % kSlots], ring[(2 * y + 1) % kSlots]);
            uint8_t* out = obuf[y & 1];
            horizontal(out);
            dma_wait(); // prefetch done, previous output row stored
            dma(addr(out), kOut + (y * kOW + x0) * kPB, OWT * kPB);
        }
        dma_wait();
    }
    roi_end();
    return 0;
}
