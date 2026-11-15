// Depthwise 3x3 convolution (YOLO26n DWConv), vector unit, W4A4.
//
//   out[y][x][c] = act( requant( bias[c] + sum_{dy,dx} w[c][dy][dx] *
//                                in[s*y+dy][s*x+dx][c] ) ),  pad 1
//
// No cross-channel reduction (K = 9), so the Zvt array has nothing to chew
// on: channels go across the vector lanes and each output is 9 vwmacc taps.
//
// Layout ("pixel-pair" int4 packing, the row-pair convention applied to two
// neighbouring pixels): a feature-map row is W/2 pixel pairs; pair k is C
// bytes, 16-byte group g holds channels 16g..16g+15 with pixel 2k in the low
// nibble and pixel 2k+1 in the high nibble. One 16-byte vle8 thus yields two
// neighbouring pixels of the same 16 channels, and both use the SAME 16-lane
// weight registers. The output is written in the same layout, so layers
// chain. Row bytes = W*C/2.
//
// Dataflow: input rows stream PSRAM -> DTCM by DMA into a ring of 4 row
// buffers (3 in use + 1 prefetching: the halo is the ring itself, every row
// is fetched once). Output rows are double-buffered and DMA'd back. Weights
// (9*C int4), the per-channel bias and requant multiplier are fetched once
// from flash and stay resident (weights unpacked once to int8).
//
// Inner loop (per output row, per 16-channel group, per input pixel pair):
// 3 packed loads (rows y-1..y+1), unpack with 3 shifts in registers (fused
// unpack, never written back), 18 vwmacc e8->e16 (int4*int4*9 fits int16),
// rolling 4 accumulators along x so each input is loaded once per output row.
// Epilogue: bias is the accumulator's initial value; requant = vmulh by a
// per-channel Q16 multiplier, clamp to [-8,7], pack the pixel pair, store.
//
//   -DH= -DW= -DC=   input shape (W even), -DSTRIDE=1|2
//   -DSILU=1         SiLU via a 256-entry int8->int4 LUT (vluxei8)
//   -DINT8=1         int8 activations (no unpack/pack): the int4 comparison
//   -DROWS=n         compute only n output rows (slice), default all
#include "ops.hpp"

using namespace ops;

#ifndef H
#define H 80
#endif
#ifndef W
#define W 80
#endif
#ifndef C
#define C 64
#endif
#ifndef STRIDE
#define STRIDE 1
#endif
#ifndef SILU
#define SILU 0
#endif
#ifndef INT8
#define INT8 0
#endif

constexpr uint32_t kOutH = (H + STRIDE - 1) / STRIDE;
constexpr uint32_t kOutW = (W + STRIDE - 1) / STRIDE;
#ifndef ROWS
#define ROWS kOutH
#endif
constexpr uint32_t kRows = ROWS;
constexpr uint32_t kGroups = C / 16;
static_assert(C % 16 == 0 && W % 4 == 0, "shape");

#if INT8
constexpr uint32_t kInRow = W * C;        // int8 NHWC
constexpr uint32_t kOutRow = kOutW * C;
constexpr uint32_t kOutBufs = 1;          // DTCM: 4 in rows + 1 out row
#else
constexpr uint32_t kInRow = W * C / 2;    // int4 pixel-pair packed
constexpr uint32_t kOutRow = kOutW * C / 2;
constexpr uint32_t kOutBufs = 2;
#endif
constexpr uint32_t kRing = STRIDE == 1 ? 4 : 5; // 3 in use + prefetch

// Flash map: weights [9][C] int4 (row-pair packed per 16-ch group: tap t in
// the low nibble, tap t+1 in the high; tap 8 pairs with a zero tap), bias
// int16 [C], requant multiplier int16 [C].
constexpr uint32_t kWeightBytes = 5 * C;
constexpr uint32_t kFlashWeights = kFlash;
constexpr uint32_t kFlashBias = kFlash + kWeightBytes;
constexpr uint32_t kFlashScale = kFlashBias + 2 * C;
constexpr uint32_t kFlashLut = kFlashScale + 2 * C;
constexpr uint32_t kRamIn = kRam;
constexpr uint32_t kRamOut = kRam + H * kInRow;

static uint8_t ring[kRing][kInRow];
static uint8_t obuf[kOutBufs][kOutRow];
static uint8_t wpacked[kWeightBytes];
static int8_t weights[10][C];             // [tap][c] int8, tap 9 = pad
static int16_t bias[C];
static int16_t scale[C];
#if SILU
static int8_t lut[256];
#endif

E2E_INLINE vint16m2_t requant(vint16m2_t acc, vint16m2_t mul) {
    vint16m2_t v = __riscv_vmulh_vv_i16m2(acc, mul, 16);
#if SILU
    v = __riscv_vmin_vx_i16m2(__riscv_vmax_vx_i16m2(v, -128, 16), 127, 16);
    return v;
#elif INT8
    return __riscv_vmin_vx_i16m2(__riscv_vmax_vx_i16m2(v, -128, 16), 127, 16);
#else
    return __riscv_vmin_vx_i16m2(__riscv_vmax_vx_i16m2(v, -8, 16), 7, 16);
#endif
}
E2E_INLINE vint8m1_t narrow(vint16m2_t v) {
    vint8m1_t q = __riscv_vncvt_x_x_w_i8m1(v, 16);
#if SILU
    q = __riscv_vluxei8_v_i8m1(lut, __riscv_vreinterpret_v_i8m1_u8m1(q), 16);
#endif
    return q;
}
// Store the output pixel pair (2j, 2j+1) of one 16-channel group.
E2E_INLINE void emit(uint8_t* orow, uint32_t j, uint32_t g, vint8m1_t q0,
                     vint8m1_t q1) {
#if INT8
    __riscv_vse8_v_i8m1(reinterpret_cast<int8_t*>(orow) + (2 * j) * C + g * 16, q0, 16);
    __riscv_vse8_v_i8m1(reinterpret_cast<int8_t*>(orow) + (2 * j + 1) * C + g * 16, q1, 16);
#else
    const vint8m1_t p = __riscv_vor_vv_i8m1(__riscv_vand_vx_i8m1(q0, 0xF, 16),
                                            __riscv_vsll_vx_i8m1(q1, 4, 16), 16);
    __riscv_vse8_v_i8m1(reinterpret_cast<int8_t*>(orow) + j * C + g * 16, p, 16);
#endif
}
// Load input pixels (2k, 2k+1) of group g from a row.
E2E_INLINE void load_pair(const uint8_t* row, uint32_t k, uint32_t g,
                          vint8m1_t& c0, vint8m1_t& c1) {
#if INT8
    const int8_t* r = reinterpret_cast<const int8_t*>(row);
    c0 = __riscv_vle8_v_i8m1(r + (2 * k) * C + g * 16, 16);
    c1 = __riscv_vle8_v_i8m1(r + (2 * k + 1) * C + g * 16, 16);
#else
    const vint8m1_t p = __riscv_vle8_v_i8m1(
        reinterpret_cast<const int8_t*>(row) + k * C + g * 16, 16);
    c0 = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(p, 4, 16), 4, 16);
    c1 = __riscv_vsra_vx_i8m1(p, 4, 16);
#endif
}
#define MAC(acc, w, x) acc = __riscv_vwmacc_vv_i16m2(acc, w, x, 16)

#define LOAD_W                                                                 \
    const vint8m1_t w0 = __riscv_vle8_v_i8m1(&weights[0][g * 16], 16);        \
    const vint8m1_t w1 = __riscv_vle8_v_i8m1(&weights[1][g * 16], 16);        \
    const vint8m1_t w2 = __riscv_vle8_v_i8m1(&weights[2][g * 16], 16);        \
    const vint8m1_t w3 = __riscv_vle8_v_i8m1(&weights[3][g * 16], 16);        \
    const vint8m1_t w4 = __riscv_vle8_v_i8m1(&weights[4][g * 16], 16);        \
    const vint8m1_t w5 = __riscv_vle8_v_i8m1(&weights[5][g * 16], 16);        \
    const vint8m1_t w6 = __riscv_vle8_v_i8m1(&weights[6][g * 16], 16);        \
    const vint8m1_t w7 = __riscv_vle8_v_i8m1(&weights[7][g * 16], 16);        \
    const vint8m1_t w8 = __riscv_vle8_v_i8m1(&weights[8][g * 16], 16);        \
    const vint16m2_t b = __riscv_vle16_v_i16m2(&bias[g * 16], 16);            \
    const vint16m2_t s = __riscv_vle16_v_i16m2(&scale[g * 16], 16)

#if STRIDE == 1
// One output row from input rows r0, r1, r2 (y-1, y, y+1).
// Input pair k (cols 2k, 2k+1) feeds out(2k-1)..out(2k+2).
#define S1_ROW(row, wl, wc, wr)                                                \
    {                                                                          \
        vint8m1_t c0, c1;                                                      \
        load_pair(row, k, g, c0, c1);                                          \
        MAC(a_m1, wr, c0);                                                     \
        MAC(a_0, wc, c0);                                                      \
        MAC(a_0, wr, c1);                                                      \
        MAC(a_p1, wl, c0);                                                     \
        MAC(a_p1, wc, c1);                                                     \
        MAC(a_p2, wl, c1);                                                     \
    }
static void row_s1(const uint8_t* r0, const uint8_t* r1, const uint8_t* r2,
                   uint8_t* orow) {
    for (uint32_t g = 0; g < kGroups; ++g) {
        LOAD_W;
        // Entering pair k: q_prev = out(2k-2) done, a_m1 = out(2k-1),
        // a_0 = out(2k) partial.
        vint8m1_t q_prev = __riscv_vmv_v_x_i8m1(0, 16);
        vint16m2_t a_m1 = b, a_0 = b;
        for (uint32_t k = 0; k < W / 2; ++k) {
            vint16m2_t a_p1 = b, a_p2 = b;
            S1_ROW(r0, w0, w1, w2);
            S1_ROW(r1, w3, w4, w5);
            S1_ROW(r2, w6, w7, w8);
            const vint8m1_t q_m1 = narrow(requant(a_m1, s));
            if (k > 0) emit(orow, k - 1, g, q_prev, q_m1);
            q_prev = narrow(requant(a_0, s));
            a_m1 = a_p1;
            a_0 = a_p2;
        }
        emit(orow, W / 2 - 1, g, q_prev, narrow(requant(a_m1, s)));
    }
}
#else
// Stride 2: out(x) uses input cols 2x-1, 2x, 2x+1 = pair x (cols 2x, 2x+1)
// plus col 2x-1 (high pixel of pair x-1), carried as a partial accumulator.
#define S2_ROW(row, wl, wc, wr)                                                \
    {                                                                          \
        vint8m1_t c0, c1;                                                      \
        load_pair(row, x, g, c0, c1);                                          \
        MAC(a, wc, c0);                                                        \
        MAC(a, wr, c1);                                                        \
        MAC(a_next, wl, c1);                                                   \
    }
static void row_s2(const uint8_t* r0, const uint8_t* r1, const uint8_t* r2,
                   uint8_t* orow) {
    for (uint32_t g = 0; g < kGroups; ++g) {
        LOAD_W;
        vint16m2_t a = b; // out(x), holds the col 2x-1 contribution
        vint8m1_t q_prev = __riscv_vmv_v_x_i8m1(0, 16);
        for (uint32_t x = 0; x < kOutW; ++x) {
            vint16m2_t a_next = b;
            S2_ROW(r0, w0, w1, w2);
            S2_ROW(r1, w3, w4, w5);
            S2_ROW(r2, w6, w7, w8);
            const vint8m1_t q = narrow(requant(a, s));
            if (x & 1) emit(orow, x >> 1, g, q_prev, q);
            q_prev = q;
            a = a_next;
        }
    }
}
#endif

E2E_INLINE void fetch_row(uint32_t r) {
    dma(kRamIn + r * kInRow, addr(ring[r % kRing]), kInRow);
}
// Padding rows: row -1 uses the ring slot that is still empty (zero) at the
// start; row H (stride 1 only) uses slot H % kRing, free by then, zeroed.
E2E_INLINE const uint8_t* in_row(int32_t r) {
    return r < 0 ? ring[kRing - 1] : ring[r % kRing];
}
static void zero_slot(uint8_t* row) {
    for (uint32_t i = 0; i < kInRow; i += 64)
        __riscv_vse8_v_u8m4(row + i, __riscv_vmv_v_x_u8m4(0, 64), 64);
}

int main() {
    roi_begin();
    // Resident parameters: one DMA each, unpack the weights once.
    dma(kFlashWeights, addr(wpacked), kWeightBytes);
    dma(kFlashBias, addr(bias), 2 * C);
    dma(kFlashScale, addr(scale), 2 * C);
#if SILU
    dma(kFlashLut, addr(lut), 256);
#endif
    fetch_row(0);
    fetch_row(1);
    dma_wait();
    for (uint32_t t = 0; t < 10; t += 2)
        unpack_i4_rows(wpacked + (t / 2) * C, &weights[t][0], C);
    mark();

    for (uint32_t y = 0; y < kRows; ++y) {
        dma_wait(); // previous output row written, next input row(s) present
        if (y > 0) {
            dma(addr(obuf[(y - 1) % kOutBufs]), kRamOut + (y - 1) * kOutRow, kOutRow);
        }
#if STRIDE == 1
        if (y + 2 < H) fetch_row(y + 2);
        if (y + 1 == H) zero_slot(ring[H % kRing]);
        row_s1(in_row(int32_t(y) - 1), in_row(y), in_row(y + 1), obuf[y % kOutBufs]);
#else
        // Output y needs input rows 2y-1..2y+1; prefetch 2y+2, 2y+3.
        if (2 * y + 3 < H) {
            fetch_row(2 * y + 2);
            fetch_row(2 * y + 3);
        } else if (2 * y + 2 < H) {
            fetch_row(2 * y + 2);
        }
        row_s2(in_row(2 * int32_t(y) - 1), in_row(2 * y), in_row(2 * y + 1),
               obuf[y % kOutBufs]);
#endif
#if INT8
        // Single output buffer (DTCM is full of int8 rows): drain it now.
        dma_wait();
#endif
    }
    dma_wait();
    dma(addr(obuf[(kRows - 1) % kOutBufs]), kRamOut + (kRows - 1) * kOutRow, kOutRow);
    dma_wait();
    roi_end();
    return 0;
}
