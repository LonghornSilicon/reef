// Max pool 3x3 stride 2 pad 1: the ResNet-18 stem, 112x112xC -> 56x56xC.
// Vector-unit operator (no matrix engine). Activations int4, NHWC, packed in
// PSRAM with the row-pair convention of ops.hpp applied per pixel: byte j of
// a pixel holds channel j (low nibble) and channel j + C/2 (high nibble).
//
// Per-nibble max straight on packed bytes (fused unpack/pack, no int8 copy):
//   high nibble: a signed int8 max of the RAW bytes orders by the high nibble
//                first, so max(raw)'s high nibble is the max of high nibbles.
//   low nibble : (b << 4) moves the low nibble up (low bits 0), so
//                max(b << 4) >>u 4 is the max of the low nibbles.
//   repack     : out = (max_raw & 0xF0) | (max_lo >>u 4).
// "Unpack" = 1 shift per input byte, "pack" = 3 ops per output byte. Pad
// bytes are 0x88 (both nibbles -8 = -inf).
//
// Dataflow: input rows stream from PSRAM into a 5-slot DTCM ring (rows
// 2y-1, 2y, 2y+1 in use, 2y+2 and 2y+3 in flight: double-buffered). Per
// output pixel the kernel loads the 2 new input pixels of each of the 3 rows
// (the third horizontal tap is the previous pixel's odd column, carried in a
// register), does the vertical and horizontal max in registers and stores the
// packed result: no intermediate rows (see maxpool_stem2p.cc for the 2-pass
// version). Output rows are double-buffered and DMA'd back to PSRAM.
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
constexpr uint32_t kPB = INT8 ? C : C / 2; // bytes per pixel (one register group)
constexpr uint32_t kSW = 2 * OWT + 1;      // slot width: left halo + 2*OWT
constexpr uint32_t kSlot = kSW * kPB;
constexpr uint32_t kSlots = 5;
constexpr uint8_t kPad = INT8 ? 0x80 : 0x88;
static_assert(kOW % OWT == 0 && kPB <= 64, "tiles");
// One pixel = one register group: m2 (32 B) for int4 C=64, m4 for int8.
#if INT8
#define LM m4
#else
#define LM m2
#endif
#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define VI CAT(CAT(vint8, LM), _t)
#define VU CAT(CAT(vuint8, LM), _t)
#define V(op, t) CAT(CAT(CAT(__riscv_, op), t), LM)

static uint8_t ring[kSlots][kSlot];
static uint8_t obuf[2][OWT * kPB];

constexpr uint32_t kIn = kRam;                // input map
constexpr uint32_t kOut = kRam + H * W * kPB; // output map

// Load input row r, columns of tile x0, into its ring slot.
E2E_INLINE void load_row(uint32_t r, uint32_t x0) {
    uint8_t* slot = ring[r % kSlots];
    if (x0 == 0) { // slot pixel 0 is the left pad (0x88); columns 0..2*OWT-1
        dma(kIn + r * W * kPB, addr(slot + kPB), 2 * OWT * kPB);
    } else {
        const uint32_t c0 = 2 * x0 - 1;
        const uint32_t n = (c0 + kSW <= W ? kSW : W - c0);
        dma(kIn + (r * W + c0) * kPB, addr(slot), n * kPB);
    }
}

#if INT8
#define LO(x)
#else
#define LO(x) x
#endif

// Vertical 3-row max of one pixel (slot byte offset p): raw and low-nibble.
E2E_INLINE void column(const int8_t* a, const int8_t* b, const int8_t* c, uint32_t p, size_t vl,
                       VI& raw, VI& lo) {
    const VI ra = V(vle8_v_,i8)(a + p, vl);
    const VI rb = V(vle8_v_,i8)(b + p, vl);
    const VI rc = V(vle8_v_,i8)(c + p, vl);
    raw = V(vmax_vv_,i8)(V(vmax_vv_,i8)(ra, rb, vl), rc, vl);
    LO(lo = V(vmax_vv_,i8)(
        V(vmax_vv_,i8)(V(vsll_vx_,i8)(ra, 4, vl), V(vsll_vx_,i8)(rb, 4, vl), vl),
        V(vsll_vx_,i8)(rc, 4, vl), vl);)
    (void)lo;
}

E2E_INLINE void row(const uint8_t* a8, const uint8_t* b8, const uint8_t* c8, uint8_t* out) {
    const auto* a = reinterpret_cast<const int8_t*>(a8);
    const auto* b = reinterpret_cast<const int8_t*>(b8);
    const auto* c = reinterpret_cast<const int8_t*>(c8);
    const size_t vl = CAT(__riscv_vsetvl_e8, LM)(kPB);
    VI pr, pl, er, el, orr, ol;
    pl = el = ol = V(vmv_v_x_,i8)(0, vl);
    column(a, b, c, 0, vl, pr, pl); // slot pixel 0: left halo / pad
    for (uint32_t x = 0; x < OWT; ++x) {
        const uint32_t p = (2 * x + 1) * kPB;
        column(a, b, c, p, vl, er, el);
        column(a, b, c, p + kPB, vl, orr, ol);
        const VI mr = V(vmax_vv_,i8)(V(vmax_vv_,i8)(pr, er, vl), orr, vl);
#if INT8
        V(vse8_v_,i8)(reinterpret_cast<int8_t*>(out + x * kPB), mr, vl);
#else
        const VI ml = V(vmax_vv_,i8)(V(vmax_vv_,i8)(pl, el, vl), ol, vl);
        const VU hi = V(vand_vx_,u8)(CAT(CAT(CAT(__riscv_vreinterpret_v_i8, LM), _u8), LM)(mr), 0xF0, vl);
        const VU lo = V(vsrl_vx_,u8)(CAT(CAT(CAT(__riscv_vreinterpret_v_i8, LM), _u8), LM)(ml), 4, vl);
        V(vse8_v_,u8)(out + x * kPB, V(vor_vv_,u8)(hi, lo, vl), vl);
#endif
        pr = orr;
        pl = ol;
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
            if (2 * y + 2 < H) load_row(2 * y + 2, x0); // prefetch the next pair
            if (2 * y + 3 < H) load_row(2 * y + 3, x0);
            const uint32_t top = (y == 0 ? 0 : 2 * y - 1); // pad row = duplicate
            uint8_t* out = obuf[y & 1];
            row(ring[top % kSlots], ring[(2 * y) % kSlots], ring[(2 * y + 1) % kSlots], out);
            dma_wait(); // prefetch landed, previous output row stored
            dma(addr(out), kOut + (y * kOW + x0) * kPB, OWT * kPB);
        }
        dma_wait();
    }
    roi_end();
    return 0;
}
