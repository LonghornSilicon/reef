// Depthwise 3x3 convolution (YOLO26n DWConv), stride 1, pad 1, W4A4.
// Vector unit, channels across the 16 lanes, hand-scheduled asm inner loop.
// (dwconv.cc is the same algorithm in intrinsics, plus the stride-2 path;
// dwconv_planar.cc puts pixels in the lanes instead.)
//
// Layout ("pixel-pair" int4 packing: the shared row-pair convention with the
// two rows being two neighbouring pixels): a feature-map row is W/2 pairs of
// C bytes; byte 16g+j of pair k holds channel 16g+j of pixel 2k (low nibble)
// and pixel 2k+1 (high nibble). One 16-byte vle8 = 2 pixels x 16 channels,
// and both pixels use the same 16-lane weight registers. Output is written in
// the same layout, so DW layers chain.
//
// Inner loop, per output row, per 16-channel group, per input pixel pair k:
//   3 x vle8 (rows y-1, y, y+1)                               LSU
//   fused unpack in registers, 2 ALU ops per load:              ALU
//     lo*16 = vsll(b, 4), hi*16 = vand(b, 0xF0)
//     (the x16 is folded into the requant multiplier; 16 * 9 * 64 fits int16)
//   18 x vwmacc.vv e8 -> e16 into 4 rolling accumulators         MUL
//     (pair k feeds outputs 2k-1 .. 2k+2, so every input byte is loaded
//      once per output row; accumulators rename by unrolling 2x)
//   epilogue for outputs 2k-1, 2k: bias is the accumulator's initial value;
//   vmulh by a per-channel Q16 multiplier, vnclip -> int8 (saturates; the
//   result is out*16 so the top nibble is the int4 output), optional SiLU
//   LUT (vluxei8, int8 Q4 pre-activation -> out*16), pack the pixel pair
//   with vsrl/vand/vor, 1 x vse8.
//
// Dataflow: input rows PSRAM -> DTCM by DMA into a ring of 4 row buffers
// (3 in use + 1 prefetching; the ring IS the halo, each row is fetched once).
// Output rows are double-buffered and DMA'd back while the next row
// computes. Weights (9C int4), bias and multiplier (4C B) are fetched once and
// stay resident; weights are unpacked once to int8.
//
//   -DH= -DW= -DC=   shape (W % 4 == 0, C % 16 == 0)
//   -DSILU=1         SiLU LUT in the epilogue
//   -DINT8=1         int8 activations (NHWC, no unpack, no pack): comparison.
//                    Still int16 accumulators, i.e. an OPTIMISTIC int8 kernel
//                    (a real int8 DW conv needs int32 accumulation).
//   -DROWS=n         compute only the first n output rows (slice)
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
#ifndef SILU
#define SILU 0
#endif
#ifndef INT8
#define INT8 0
#endif
#ifndef ROWS
#define ROWS H
#endif
constexpr uint32_t kRows = ROWS;
constexpr uint32_t kGroups = C / 16;
static_assert(C % 16 == 0 && W % 4 == 0, "shape");

#if INT8
constexpr uint32_t kPairBytes = 2 * C;  // two int8 pixels
#else
constexpr uint32_t kPairBytes = C;      // two int4 pixels
#endif
constexpr uint32_t kRowData = W / 2 * kPairBytes;      // bytes in PSRAM
// DTCM rows carry 2 zero pairs after the data (right padding + the junk
// iteration of the unrolled loop); output rows carry 1 slack pair before
// (the k = 0 junk store) and 2 after.
constexpr uint32_t kRowBuf = kRowData + 2 * kPairBytes;
constexpr uint32_t kOutBuf = kRowData + 3 * kPairBytes;
constexpr uint32_t kRing = 4;
constexpr uint32_t kOutBufs = INT8 ? 1 : 2; // int8 rows: DTCM too small for 2

// Flash: weights [5][C] (row-pair int4: taps 2i/2i+1), bias int16 [C]
// (pre-scaled x16), multiplier int16 [C], SiLU LUT [256].
constexpr uint32_t kFlashWeights = kFlash;
constexpr uint32_t kFlashBias = kFlash + 5 * C;
constexpr uint32_t kFlashScale = kFlashBias + 2 * C;
constexpr uint32_t kFlashLut = kFlashScale + 2 * C;
constexpr uint32_t kRamIn = kRam;
constexpr uint32_t kRamOut = kRam + H * kRowData;

static uint8_t ring[kRing][kRowBuf];
static uint8_t obuf[kOutBufs][kOutBuf];
static uint8_t wpacked[5 * C];
static int8_t weights[10][C]; // [tap][channel], tap 9 unused
static int16_t bias[C];
static int16_t scale[C];
static int8_t lut[256];

// ---- inner loop -----------------------------------------------------------
// Registers: v23..v31 taps (dy*3+dx), v20 bias, v18 multiplier (e16m2),
// v8 v10 v12 v14 rolling accumulators (e16m2), v16 epilogue temp,
// v2-v4 / v0 v1 v5 row inputs, v6/v7 the held output of the previous half.
#define STR_(x) #x
#define STR(x) STR_(x)
// Software-pipelined by one row: the load of the next row (and, from row 2,
// of the next pair's row 0) is issued before the current row's MACs, and the
// MACs alternate accumulators so no two dependent vwmacc are adjacent (a
// dependent vwmacc costs 4 cycles, an independent one ~1).
// Row inputs: L0..L2 = v0..v2 (packed load, unpacked in place to hi*16),
// X0..X2 = v3..v5 (lo*16).
#if INT8
#define LOAD(R, L, X)                                                          \
    "vle8.v " X ", (" R ")\n\t"                                                \
    "add t2, " R ", %[c]\n\t"                                                  \
    "vle8.v " L ", (t2)\n\t"                                                   \
    "add " R ", " R ", %[pair]\n\t"
#define UNPACK(L, X) ""
#else
#define LOAD(R, L, X)                                                          \
    "vle8.v " L ", (" R ")\n\t"                                                \
    "add " R ", " R ", %[pair]\n\t"
#define UNPACK(L, X)                                                           \
    "vsll.vi " X ", " L ", 4\n\t"                                              \
    "vand.vi " L ", " L ", -16\n\t"
#endif
// X = pixel 2k of this row, L = pixel 2k+1.
#define MACS(X, L, WL, WC, WR, A, B, CC, D)                                    \
    "vwmacc.vv " A ", " WR ", " X "\n\t"                                       \
    "vwmacc.vv " B ", " WC ", " X "\n\t"                                       \
    "vwmacc.vv " CC ", " WL ", " X "\n\t"                                      \
    "vwmacc.vv " B ", " WR ", " L "\n\t"                                       \
    "vwmacc.vv " D ", " WL ", " L "\n\t"                                       \
    "vwmacc.vv " CC ", " WC ", " L "\n\t"
#if SILU
#define LUT2(QA, QB)                                                           \
    "vluxei8.v " QA ", (%[lut]), " QA "\n\tvluxei8.v " QB ", (%[lut]), " QB "\n\t"
#else
#define LUT2(QA, QB) ""
#endif
#if INT8
// Outputs 2k-2 (held in QIN) and 2k-1 (v16) as int8 pixels.
#define STORE(QIN)                                                             \
    "vse8.v " QIN ", (%[out])\n\t"                                             \
    "add t2, %[out], %[c]\n\t"                                                 \
    "vse8.v v16, (t2)\n\t"
#else
#define STORE(QIN)                                                             \
    "vsrl.vi " QIN ", " QIN ", 4\n\t"                                          \
    "vand.vi v16, v16, -16\n\t"                                                \
    "vor.vv v16, v16, " QIN "\n\t"                                             \
    "vse8.v v16, (%[out])\n\t"
#endif
// One input pair. Entering: A = out(2k-1), B = out(2k) partial, QIN =
// out(2k-2) requantized, v0 = row y-1 of pair k already loaded. Leaving:
// CC = out(2k+1), D = out(2k+2), QOUT = out(2k), v0 = row y-1 of pair k+1.
#define HALF(A, B, CC, D, QIN, QOUT)                                           \
    "vsetivli zero, 16, e8, m1, ta, ma\n\t"                                    \
    UNPACK("v0", "v3")                                                         \
    LOAD("%[r1]", "v1", "v4")                                                  \
    "vmv2r.v " CC ", v20\n\t"                                                  \
    "vmv2r.v " D ", v20\n\t"                                                   \
    MACS("v3", "v0", "v23", "v24", "v25", A, B, CC, D)                         \
    UNPACK("v1", "v4")                                                         \
    LOAD("%[r2]", "v2", "v5")                                                  \
    MACS("v4", "v1", "v26", "v27", "v28", A, B, CC, D)                         \
    UNPACK("v2", "v5")                                                         \
    LOAD("%[r0]", "v0", "v3")                                                  \
    MACS("v5", "v2", "v29", "v30", "v31", A, B, CC, D)                         \
    "vsetivli zero, 16, e16, m2, ta, ma\n\t"                                   \
    "vmulh.vv v16, " A ", v18\n\t"                                             \
    "vmulh.vv " A ", " B ", v18\n\t"                                           \
    "vsetivli zero, 16, e8, m1, ta, ma\n\t"                                    \
    "vnclip.wi v16, v16, 0\n\t"                                                \
    "vnclip.wi " QOUT ", " A ", 0\n\t" LUT2("v16", QOUT) STORE(QIN)            \
    "add %[out], %[out], %[pair]\n\t"

// One output row of one 16-channel group. r0..r2 point at pair 0 of the group
// in rows y-1, y, y+1; out at the slack pair before pair 0.
static void row_group(const uint8_t* r0, const uint8_t* r1, const uint8_t* r2,
                      uint8_t* out, uint32_t g) {
    const int8_t* w = &weights[0][g * 16];
    const int16_t* b = &bias[g * 16];
    const int16_t* s = &scale[g * 16];
    uint32_t n = W / 2 + 1; // pairs 0 .. W/2 (the last one is zero padding)
    asm volatile(
        "vsetivli zero, 16, e8, m1, ta, ma\n\t"
        "mv t2, %[w]\n\t"
        "vle8.v v23, (t2)\n\tadd t2, t2, %[c]\n\t"
        "vle8.v v24, (t2)\n\tadd t2, t2, %[c]\n\t"
        "vle8.v v25, (t2)\n\tadd t2, t2, %[c]\n\t"
        "vle8.v v26, (t2)\n\tadd t2, t2, %[c]\n\t"
        "vle8.v v27, (t2)\n\tadd t2, t2, %[c]\n\t"
        "vle8.v v28, (t2)\n\tadd t2, t2, %[c]\n\t"
        "vle8.v v29, (t2)\n\tadd t2, t2, %[c]\n\t"
        "vle8.v v30, (t2)\n\tadd t2, t2, %[c]\n\t"
        "vle8.v v31, (t2)\n\t"
        "vsetivli zero, 16, e16, m2, ta, ma\n\t"
        "vle16.v v20, (%[b])\n\t"
        "vle16.v v18, (%[s])\n\t"
        "vmv2r.v v8, v20\n\t"
        "vmv2r.v v10, v20\n\t"
        "vsetivli zero, 16, e8, m1, ta, ma\n\t"
        LOAD("%[r0]", "v0", "v3")
        "srli t3, %[n], 1\n\t"
        "1:\n\t"
        HALF("v8", "v10", "v12", "v14", "v7", "v6")
        HALF("v12", "v14", "v8", "v10", "v6", "v7")
        "addi t3, t3, -1\n\t"
        "bnez t3, 1b\n\t"
        "andi t3, %[n], 1\n\t"
        "beqz t3, 2f\n\t"
        HALF("v8", "v10", "v12", "v14", "v7", "v6")
        "2:\n\t"
        : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), [out] "+r"(out)
        : [w] "r"(w), [b] "r"(b), [s] "r"(s), [n] "r"(n), [c] "r"(C),
          [pair] "r"(kPairBytes), [lut] "r"(lut)
        : "t2", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
          "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
          "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
          "v29", "v30", "v31", "vl", "vtype", "memory");
}

// ---- row streaming ----------------------------------------------------------
E2E_INLINE void fetch_row(uint32_t r) {
    dma(kRamIn + r * kRowData, addr(ring[r % kRing]), kRowData);
}
// Row -1 reads ring slot 3 (still all zero at the start); row H reads slot
// H % 4, which is free by then and zeroed once.
E2E_INLINE const uint8_t* in_row(int32_t r) {
    return r < 0 ? ring[kRing - 1] : ring[r % kRing];
}
static void zero_slot(uint8_t* row) {
    for (uint32_t i = 0; i < kRowData; i += 64)
        __riscv_vse8_v_u8m4(row + i, __riscv_vmv_v_x_u8m4(0, 64), 64);
}

int main() {
    roi_begin();
    dma(kFlashWeights, addr(wpacked), 5 * C);
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
        dma_wait(); // row y+1 present, output row y-1's buffer drained
        if (y > 0) {
            dma(addr(obuf[(y - 1) % kOutBufs] + kPairBytes),
                kRamOut + (y - 1) * kRowData, kRowData);
        }
        if (y + 2 < H) fetch_row(y + 2);
        if (y + 1 == H) zero_slot(ring[H % kRing]);
#if INT8
        if (y > 0) dma_wait(); // single output buffer
#endif
        const uint8_t* r0 = in_row(int32_t(y) - 1);
        const uint8_t* r1 = in_row(y);
        const uint8_t* r2 = in_row(y + 1);
        uint8_t* o = obuf[y % kOutBufs];
        for (uint32_t g = 0; g < kGroups; ++g)
            row_group(r0 + g * 16, r1 + g * 16, r2 + g * 16, o + g * 16, g);
    }
    dma_wait();
    dma(addr(obuf[(kRows - 1) % kOutBufs] + kPairBytes),
        kRamOut + (kRows - 1) * kRowData, kRowData);
    dma_wait();
    roi_end();
    return 0;
}
