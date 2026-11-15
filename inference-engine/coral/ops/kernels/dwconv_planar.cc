// Depthwise 3x3 convolution (YOLO26n DWConv), vector unit, W4A4,
// CHANNEL-PLANAR variant: pixels across the vector lanes, one channel at a
// time, so each tap is a scalar weight: vwmacc.vx at e8m4 -> e16m8.
//
// Why: on this core every vector instruction holds one of the 8 scalar ROB
// entries until it completes, so throughput is set by work per instruction.
// The channels-in-lanes kernel (dwconv.cc) needs 9 per-channel weight
// vectors resident and is stuck at m1 (16 lanes); here weights are scalars
// and the MACs run 40-64 lanes wide.
//
// Layout (channel-planar, row-pair int4 packing = the shared convention with
// the two rows being image rows r, r+1): plane c is H/2 pair-rows of W bytes;
// byte x of pair-row p holds pixel (2p, x) low and (2p+1, x) high. This is
// also the [K][pixels] operand layout the Zvt conv/linear kernels consume.
// Output is written in the same layout (2 output rows per packed vector).
//
// Per output pair-row (rows y, y+1), per strip of up to 40 pixels:
//   rows y-1..y+2 come from 3 packed pair-rows (hi of p-1, both of p, lo of
//   p+1); each is loaded at x-1, x, x+1 (9 unaligned vle8, fused unpack with
//   vsll/vsra in registers), 18 vwmacc.vx into 2 int16 accumulators (bias as
//   the initial value), then vmulh.vx requant, clamp [-8,7], pack, 1 store.
// Zero padding: left/right by masking the edge lanes of the dx=-1/+1 MACs,
// top/bottom by a zero pair-row.
//
// Dataflow: planes are contiguous in PSRAM, so a chunk of whole channels is
// one DMA; chunks (about 3 KB) are double-buffered in and out. The halo is
// free (the whole plane is in DTCM). Weights/bias/scale (9C int4 + 4C B) are
// fetched once and stay resident.
//
//   -DH= -DW= -DC=   shape (H even, W <= 80), -DSILU=1, -DINT8=1, -DCH=n slice
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
#ifndef CH
#define CH C // channels computed (slice)
#endif

#if INT8
constexpr uint32_t kPlane = H * W;      // int8 planar
#else
constexpr uint32_t kPlane = H / 2 * W;  // int4 row-pair packed
#endif
constexpr uint32_t kChunkTarget = 3200;
constexpr uint32_t kChunkCh = kPlane >= kChunkTarget ? 1 : kChunkTarget / kPlane;
constexpr uint32_t kChunks = (CH + kChunkCh - 1) / kChunkCh;
constexpr uint32_t kStrip = W > 64 ? W / 2 : W; // 80 -> 2 x 40
constexpr uint32_t kStrips = W / kStrip;
static_assert(H % 2 == 0 && W % kStrip == 0 && kStrip <= 64, "shape");

constexpr uint32_t kFlashWeights = kFlash;              // [C][9] int4 (5 B/ch)
constexpr uint32_t kFlashBias = kFlash + 5 * C;         // int16 [C]
constexpr uint32_t kFlashScale = kFlashBias + 2 * C;    // int16 [C]
constexpr uint32_t kFlashLut = kFlashScale + 2 * C;
constexpr uint32_t kRamIn = kRam;
constexpr uint32_t kRamOut = kRam + C * kPlane;

static uint8_t inbuf[2][kChunkCh * kPlane + 16];
static uint8_t outbuf[2][kChunkCh * kPlane];
static uint8_t zero_row[W + 16];
static uint8_t wpacked[5 * C];
static int8_t weights[C][10];
static int16_t bias[C];
static int16_t scale[C];
#if SILU
static int8_t lut[256];
#endif

// The strip is hand-scheduled asm: the compiler spills the m8 accumulators
// once v0 is taken by a mask. Registers: v8-v15 = acc row y, v16-v23 = acc
// row y+1 (e16m8), v24-v27 load, v28-v31 / v4-v7 unpacked rows (e8m4), v0
// edge mask. vwmacc.vx runs at e8m4 (source SEW) into e16m8.
struct Params {
    int32_t w[9];
    int32_t bias, scale;
};
static uint8_t mask_left[16], mask_right[16];

#define M_NONE ""
#define M_ON ", v0.t"
#define LDMASK(m) "vlm.v v0, (" m ")\n\t"
// One dx column: input pointers pa/pb/pc offset by OFF; weights w[d], w[3+d],
// w[6+d] in registers %[wa] %[wb] %[wc].
#if INT8
#define DX_ASM(OFF, WA, WB, WC, MK)                                            \
    "vle8.v v24, (%[pa])\n\t"                                                  \
    "vwmacc.vx v8, " WA ", v24" MK "\n\t"                                      \
    "vle8.v v28, (%[pb])\n\t"                                                  \
    "vwmacc.vx v8, " WB ", v28" MK "\n\t"                                      \
    "vwmacc.vx v16, " WA ", v28" MK "\n\t"                                     \
    "vle8.v v4, (%[pc])\n\t"                                                   \
    "vwmacc.vx v8, " WC ", v4" MK "\n\t"                                       \
    "vwmacc.vx v16, " WB ", v4" MK "\n\t"                                      \
    "vle8.v v24, (%[pd])\n\t"                                                  \
    "vwmacc.vx v16, " WC ", v24" MK "\n\t"                                     \
    "addi %[pa], %[pa], 1\n\taddi %[pb], %[pb], 1\n\t"                         \
    "addi %[pc], %[pc], 1\n\taddi %[pd], %[pd], 1\n\t"
#else
#define DX_ASM(OFF, WA, WB, WC, MK)                                            \
    "vle8.v v24, (%[pa])\n\t"                                                  \
    "vsra.vi v28, v24, 4\n\t"                                                  \
    "vwmacc.vx v8, " WA ", v28" MK "\n\t"                                      \
    "vle8.v v24, (%[pb])\n\t"                                                  \
    "vsll.vi v28, v24, 4\n\t"                                                  \
    "vsra.vi v4, v24, 4\n\t"                                                   \
    "vsra.vi v28, v28, 4\n\t"                                                  \
    "vwmacc.vx v8, " WB ", v28" MK "\n\t"                                      \
    "vwmacc.vx v16, " WA ", v28" MK "\n\t"                                     \
    "vwmacc.vx v8, " WC ", v4" MK "\n\t"                                       \
    "vwmacc.vx v16, " WB ", v4" MK "\n\t"                                      \
    "vle8.v v24, (%[pc])\n\t"                                                  \
    "vsll.vi v28, v24, 4\n\t"                                                  \
    "vsra.vi v28, v28, 4\n\t"                                                  \
    "vwmacc.vx v16, " WC ", v28" MK "\n\t"                                     \
    "addi %[pa], %[pa], 1\n\taddi %[pb], %[pb], 1\n\taddi %[pc], %[pc], 1\n\t"
#endif
#if SILU
#define CLAMP_LO "-128"
#define CLAMP_HI "127"
#define LUT_ASM "vluxei8.v v24, (%[lut]), v24\n\tvluxei8.v v28, (%[lut]), v28\n\t"
#elif INT8
#define CLAMP_LO "-128"
#define CLAMP_HI "127"
#define LUT_ASM ""
#else
#define CLAMP_LO "-8"
#define CLAMP_HI "7"
#define LUT_ASM ""
#endif
#if INT8
#define STORE_ASM                                                              \
    "vse8.v v24, (%[out])\n\tadd %[out], %[out], %[w]\n\tvse8.v v28, (%[out])\n\t"
#else
#define STORE_ASM                                                              \
    "vand.vi v24, v24, 15\n\tvsll.vi v28, v28, 4\n\tvor.vv v24, v24, v28\n\t"  \
    "vse8.v v24, (%[out])\n\t"
#endif
#define STRIP_ASM(PRE_L, MK_L, PRE_R, MK_R)                                    \
    asm volatile(                                                              \
        "li t0, " CLAMP_LO "\n\tli t1, " CLAMP_HI "\n\t"                       \
        "vsetvli zero, %[vl], e16, m8, ta, mu\n\t"                             \
        "vmv.v.x v8, %[b]\n\tvmv.v.x v16, %[b]\n\t"                            \
        "vsetvli zero, %[vl], e8, m4, ta, mu\n\t" PRE_L                        \
        "addi %[pa], %[pa], -1\n\taddi %[pb], %[pb], -1\n\t"                   \
        "addi %[pc], %[pc], -1\n\t" PD_DEC                                     \
        DX_ASM(-1, "%[w0]", "%[w3]", "%[w6]", MK_L)                            \
        DX_ASM(0, "%[w1]", "%[w4]", "%[w7]", M_NONE) PRE_R                     \
        DX_ASM(1, "%[w2]", "%[w5]", "%[w8]", MK_R)                             \
        "vsetvli zero, %[vl], e16, m8, ta, ma\n\t"                             \
        "vmulh.vx v8, v8, %[s]\n\tvmulh.vx v16, v16, %[s]\n\t"                 \
        "vmax.vx v8, v8, t0\n\tvmax.vx v16, v16, t0\n\t"                       \
        "vmin.vx v8, v8, t1\n\tvmin.vx v16, v16, t1\n\t"                       \
        "vsetvli zero, %[vl], e8, m4, ta, ma\n\t"                              \
        "vnsrl.wi v24, v8, 0\n\tvnsrl.wi v28, v16, 0\n\t" LUT_ASM STORE_ASM    \
        : [pa] "+r"(pa), [pb] "+r"(pb), [pc] "+r"(pc), [pd] "+r"(pd),          \
          [out] "+r"(out)                                                      \
        : [vl] "r"(kStrip), [b] "r"(p.bias), [s] "r"(p.scale),                 \
          [w0] "r"(p.w[0]), [w1] "r"(p.w[1]), [w2] "r"(p.w[2]),                \
          [w3] "r"(p.w[3]), [w4] "r"(p.w[4]), [w5] "r"(p.w[5]),                \
          [w6] "r"(p.w[6]), [w7] "r"(p.w[7]), [w8] "r"(p.w[8]),                \
          [ml] "r"(mask_left), [mr] "r"(mask_right), [lut] "r"(LUT_PTR),       \
          [w] "r"(W)                                                           \
        : "t0", "t1", "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",  \
          "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",       \
          "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",       \
          "v30", "v31", "vl", "vtype", "memory")
#if INT8
#define PD_DEC "addi %[pd], %[pd], -1\n\t"
#else
#define PD_DEC ""
#endif
#if SILU
#define LUT_PTR lut
#else
#define LUT_PTR nullptr
#endif

// int4: pa = pair-row (y-2, y-1), pb = (y, y+1), pc = (y+2, y+3).
// int8: pa..pd = rows y-1..y+2.
template <bool kLeft, bool kRight>
E2E_INLINE void strip(const uint8_t* pa, const uint8_t* pb, const uint8_t* pc,
                      const uint8_t* pd, const Params& p, uint8_t* out) {
    if (kLeft && kRight) {
        STRIP_ASM(LDMASK("%[ml]"), M_ON, LDMASK("%[mr]"), M_ON);
    } else if (kLeft) {
        STRIP_ASM(LDMASK("%[ml]"), M_ON, "", M_NONE);
    } else if (kRight) {
        STRIP_ASM("", M_NONE, LDMASK("%[mr]"), M_ON);
    } else {
        STRIP_ASM("", M_NONE, "", M_NONE);
    }
}

static void channel(const uint8_t* plane, uint8_t* oplane, uint32_t c) {
    Params p;
    for (uint32_t t = 0; t < 9; ++t) p.w[t] = weights[c][t];
    p.bias = bias[c];
    p.scale = scale[c];
#if INT8
    for (uint32_t y = 0; y < H; y += 2) {
        const uint8_t* rm1 = y > 0 ? plane + (y - 1) * W : zero_row;
        const uint8_t* r2 = y + 2 < H ? plane + (y + 2) * W : zero_row;
        const uint8_t* r0 = plane + y * W;
        uint8_t* out = oplane + y * W;
        if (kStrips == 1) {
            strip<true, true>(rm1, r0, r0 + W, r2, p, out);
        } else {
            strip<true, false>(rm1, r0, r0 + W, r2, p, out);
            strip<false, true>(rm1 + kStrip, r0 + kStrip, r0 + W + kStrip,
                               r2 + kStrip, p, out + kStrip);
        }
    }
#else
    for (uint32_t pr = 0; pr < H / 2; ++pr) {
        const uint8_t* pa = pr > 0 ? plane + (pr - 1) * W : zero_row;
        const uint8_t* pb = plane + pr * W;
        const uint8_t* pc = pr + 1 < H / 2 ? plane + (pr + 1) * W : zero_row;
        uint8_t* out = oplane + pr * W;
        if (kStrips == 1) {
            strip<true, true>(pa, pb, pc, nullptr, p, out);
        } else {
            strip<true, false>(pa, pb, pc, nullptr, p, out);
            strip<false, true>(pa + kStrip, pb + kStrip, pc + kStrip, nullptr, p,
                               out + kStrip);
        }
    }
#endif
}

E2E_INLINE uint32_t chunk_ch(uint32_t i) {
    return (i + 1) * kChunkCh <= CH ? kChunkCh : CH - i * kChunkCh;
}

int main() {
    roi_begin();
    dma(kFlashWeights, addr(wpacked), 5 * C);
    dma(kFlashBias, addr(bias), 2 * C);
    dma(kFlashScale, addr(scale), 2 * C);
#if SILU
    dma(kFlashLut, addr(lut), 256);
#endif
    dma(kRamIn, addr(inbuf[0] + 16), chunk_ch(0) * kPlane);
    dma_wait();
    // Unpack the 9 taps of every channel once (resident, int8 scalars).
    for (uint32_t c = 0; c < C; ++c) {
        for (uint32_t t = 0; t < 9; t += 2) {
            const uint8_t byte = wpacked[c * 5 + t / 2];
            weights[c][t] = int8_t(byte << 4) >> 4;
            weights[c][t + 1] = int8_t(byte) >> 4;
        }
    }
    // Edge masks: lane 0 of the dx=-1 vector is x=-1 (left strip); the last
    // lane of the dx=+1 vector is x=W (right strip).
    {
        const vuint8m4_t lane = __riscv_vid_v_u8m4(kStrip);
        __riscv_vsm_v_b2(mask_left, __riscv_vmsne_vx_u8m4_b2(lane, 0, kStrip), kStrip);
        __riscv_vsm_v_b2(mask_right, __riscv_vmsne_vx_u8m4_b2(lane, kStrip - 1, kStrip),
                         kStrip);
    }
    mark();

    for (uint32_t i = 0; i < kChunks; ++i) {
        dma_wait(); // chunk i present, output chunk i-1's buffer free
        if (i > 0) {
            dma(addr(outbuf[(i - 1) & 1]), kRamOut + (i - 1) * kChunkCh * kPlane,
                chunk_ch(i - 1) * kPlane);
        }
        if (i + 1 < kChunks) {
            dma(kRamIn + (i + 1) * kChunkCh * kPlane, addr(inbuf[(i + 1) & 1] + 16),
                chunk_ch(i + 1) * kPlane);
        }
        for (uint32_t j = 0; j < chunk_ch(i); ++j) {
            channel(inbuf[i & 1] + 16 + j * kPlane, outbuf[i & 1] + j * kPlane,
                    i * kChunkCh + j);
        }
    }
    dma_wait();
    dma(addr(outbuf[(kChunks - 1) & 1]), kRamOut + (kChunks - 1) * kChunkCh * kPlane,
        chunk_ch(kChunks - 1) * kPlane);
    dma_wait();
    roi_end();
    return 0;
}
