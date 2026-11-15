// Conv2D (dense, BatchNorm folded) on the Zvt matrix engine: W4A4 implicit
// GEMM, int32 accumulate, fused bias + requantize + optional ReLU, output
// packed int4. No im2col in external memory.
//
// Dataflow (see coral/ops/results/conv2d.md):
//  * Activations are planar per channel PAIR (row-pair nibble packing: byte j
//    = pixel j of channel 2c in the low nibble, channel 2c+1 in the high), on
//    a zero-padded grid of width Wg, flattened. Output pixel p = y*Wg + x and
//    tap (ky,kx) read input p + off(ky,kx): for 16 consecutive output pixels
//    the A row of one channel is 16 CONTIGUOUS bytes, so a tap is a plain
//    GEMM with a shifted base pointer. The Wg-W "junk" columns per row are
//    computed and zeroed; they land exactly on the next layer's pad columns.
//    Stride 2 uses 4 phase planes (even/odd rows x even/odd columns), which
//    makes every tap contiguous again.
//  * Register blocking 2x2: 2 pixel tiles (32 px, tm = 16 each) x 2 groups of
//    16 output channels = the 4 int32 accumulators. Per k-step (4 input
//    channels): A = 2 packed loads per tile, B = 2 packed loads per group,
//    unpacked in registers (vsll/vsra) -> 8 loads for 4 vtmm (mac4: 20).
//  * Loop order: strip (NG pixel groups, rolling halo) > group of 32 px >
//    32-channel output block > tap/K-chunk. Weights are streamed from flash
//    per pixel group (double-buffered chunks) unless the whole layer fits the
//    DTCM, in which case they are loaded once (resident).
//  * The input strip is double-buffered; only the new G bytes per plane are
//    DMAed, the halo is copied from the previous strip's tail in DTCM.
//
// Shapes: -DCIN -DCOUT -DH -DW -DKH -DKW -DSTRIDE -DPAD -DRELU, -DGROUPS=n
// runs only the first n pixel groups (a slice). -DINT8=1: int8 everything.
#include "ops.hpp"

using namespace ops;

#ifndef CIN
#define CIN 128
#endif
#ifndef COUT
#define COUT 128
#endif
#ifndef H
#define H 28
#endif
#ifndef W
#define W 28
#endif
#ifndef KH
#define KH 3
#endif
#ifndef KW
#define KW 3
#endif
#ifndef STRIDE
#define STRIDE 1
#endif
#ifndef PAD
#define PAD 1
#endif
#ifndef RELU
#define RELU 1
#endif
#ifndef INT8
#define INT8 0
#endif
#ifndef GROUPS
#define GROUPS 0 // 0: the whole layer
#endif
#ifndef RES_ALLOW
#define RES_ALLOW 1 // allow resident weights when they fit
#endif

// ---- geometry --------------------------------------------------------------
constexpr uint32_t r16(uint32_t v) { return (v + 15) & ~15U; }
constexpr uint32_t r64(uint32_t v) { return (v + 63) & ~63U; }
constexpr uint32_t cdiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

constexpr uint32_t kCin = (CIN + 3) / 4 * 4; // padded to the k-step
constexpr uint32_t kCout = COUT;
static_assert(kCout % 32 == 0, "COUT must be a multiple of 32");
constexpr uint32_t kS = STRIDE;
constexpr uint32_t kHp = H + 2 * PAD, kWp = W + 2 * PAD;
constexpr uint32_t kHo = (kHp - KH) / kS + 1, kWo = (kWp - KW) / kS + 1;
constexpr uint32_t kPhases = kS * kS;
constexpr uint32_t kWg = kS == 1 ? kWp : cdiv(kWp, 2); // flat grid width
constexpr uint32_t kHalo =
    kS == 1 ? (KH - 1) * kWp + (KW - 1) : ((KH - 1) / 2) * kWg + (KW - 1) / 2;
constexpr uint32_t kP = kHo * kWg; // flat output pixels (incl. junk columns)
constexpr uint32_t kTaps = KH * KW;
constexpr uint32_t kGroupsAll = cdiv(kP, 32);
constexpr uint32_t kGroups = GROUPS ? GROUPS : kGroupsAll;

// int4: one plane holds a channel pair; int8: one channel.
constexpr uint32_t kChPerPlane = INT8 ? 1 : 2;
constexpr uint32_t kInPlanes = kCin / kChPerPlane;
constexpr uint32_t kOutPlanes = kCout / kChPerPlane;
// Weight chunk: KC input channels of one tap for one 32-channel block,
// <= 2 KB. Row layout per k-step: int4 [pair01 g0|pair01 g1|pair23 g0|pair23
// g1] (64 B), int8 [row0 g0|row0 g1|...|row3 g1] (128 B).
constexpr uint32_t kStepBytes = INT8 ? 128 : 64;
constexpr uint32_t kKc = (kCin / 4 * kStepBytes <= 2048) ? kCin : 2048 / kStepBytes * 4;
static_assert(kCin % kKc == 0, "K chunking");
constexpr uint32_t kChunk = kKc / 4 * kStepBytes;
constexpr uint32_t kKChunks = kCin / kKc;
constexpr uint32_t kOcBlocks = kCout / 32;
constexpr uint32_t kChunksPerBlock = kTaps * kKChunks;
constexpr uint32_t kChunksPerGroup = kOcBlocks * kChunksPerBlock;
constexpr uint32_t kWeightBytes = kChunksPerGroup * kChunk;

// ---- DTCM budget -------------------------------------------------------------
constexpr uint32_t kAvail = 32768 - 1024 - 640; // stack + small statics
constexpr uint32_t kParams = kCout * 8;
constexpr uint32_t kWB(uint32_t ng) { return r16(ng * 32 + kHalo); }
constexpr uint32_t strip_bytes(uint32_t ng, uint32_t bufs) {
    return bufs * kInPlanes * kPhases * kWB(ng);
}
constexpr uint32_t kObuf1 = kOutPlanes * 32;
// Resident weights if they and a 1-group double-buffered strip fit.
constexpr bool kResident = RES_ALLOW && kWeightBytes + 2 * kObuf1 + kParams +
                                            strip_bytes(1, 2) <= kAvail;
constexpr uint32_t kWbuf = kResident ? kWeightBytes : 2 * kChunk;
constexpr uint32_t kObufs = (kWbuf + 2 * kObuf1 + kParams + strip_bytes(1, 1) <= kAvail) ? 2 : 1;
constexpr uint32_t kFixed = kWbuf + kObufs * kObuf1 + kParams;
constexpr uint32_t kStripBufs = kFixed + strip_bytes(1, 2) <= kAvail ? 2 : 1;
constexpr uint32_t pick_ng() {
    uint32_t ng = 1;
    while (ng < 16 && ng < kGroups && kFixed + strip_bytes(ng + 1, kStripBufs) <= kAvail) ++ng;
    return ng;
}
constexpr uint32_t kNG = pick_ng();
static_assert(kFixed + strip_bytes(kNG, kStripBufs) <= kAvail, "DTCM overflow");
constexpr uint32_t kG = kNG * 32;            // flat pixels per strip
constexpr uint32_t kWin = kWB(kNG);          // window bytes per plane-phase
constexpr uint32_t kPlaneStride = kPhases * kWin;
constexpr uint32_t kStripBuf = kInPlanes * kPlaneStride;
constexpr uint32_t kStrips = cdiv(kGroups, kNG);

// ---- external memory map ----------------------------------------------------
constexpr uint32_t kInPhase = r64(kStrips * kG + kHalo + 64); // per plane-phase
constexpr uint32_t kIn = kRam;
constexpr uint32_t kOut = kRam + (4u << 20);
constexpr uint32_t kOutPlane = r64(kP + 2 * kWp + 64);
constexpr uint32_t kOutBase = (kS == 1 && PAD) ? kWg + 1 : 0; // next layer's pad
constexpr uint32_t kWeights = kFlash;
constexpr uint32_t kParamsFlash = kFlash + kWeightBytes;

// ---- DTCM buffers -------------------------------------------------------------
static uint8_t sbuf[kStripBufs][kStripBuf] __attribute__((aligned(16)));
static uint8_t wbuf[kWbuf] __attribute__((aligned(16)));
static uint8_t obuf[kObufs][kObuf1] __attribute__((aligned(16)));
static int32_t params[kCout][2]; // bias, multiplier (Q31)
static uint16_t tap_off[kTaps];
constexpr uint32_t kShift = 6;

// ---- DMA job streams (spread over the chunk iterations) --------------------
static uint32_t in_j = 0, in_end = 0, in_strip = 0, in_dst = 0;
static uint32_t out_j = 0, out_end = 0, out_group = 0, out_src = 0;
constexpr uint32_t kInJobs = kInPlanes * kPhases;
constexpr uint32_t kChunksPerStrip = kNG * kChunksPerGroup;
constexpr uint32_t kInPerChunk = cdiv(kInJobs, kChunksPerStrip);
constexpr uint32_t kOutPerChunk = cdiv(kOutPlanes, kChunksPerGroup);

static void issue_in(uint32_t n) {
    const uint32_t off = in_strip ? in_strip * kG + kHalo : 0;
    const uint32_t len = in_strip ? kG : kG + kHalo;
    const uint32_t doff = in_strip ? kHalo : 0;
    for (; n && in_j < in_end; --n, ++in_j) {
        dma(kIn + in_j * kInPhase + off, in_dst + in_j * kWin + doff, len);
    }
}
static void issue_out(uint32_t n) {
    const uint32_t p0 = out_group * 32;
    const uint32_t len = kP - p0 < 32 ? kP - p0 : 32;
    for (; n && out_j < out_end; --n, ++out_j) {
        dma(out_src + out_j * 32, kOut + out_j * kOutPlane + kOutBase + p0, len);
    }
}

// ---- inner loop: 2 pixel tiles x 2 channel groups, `steps` k-steps ----------
// A tile 0 -> v8.., tile 1 -> v24..; B group 0 -> v16.., group 1 -> v0..
// mt0 = t0 x g0, mt4 = t0 x g1, mt8 = t1 x g0, mt12 = t1 x g1.
#define VTMM4                                                                  \
    ".insn r 0x77, 0, 0x79, x1, x16, x8\n\t"                                   \
    ".insn r 0x77, 0, 0x79, x9, x0, x8\n\t"                                    \
    ".insn r 0x77, 0, 0x79, x17, x16, x24\n\t"                                 \
    ".insn r 0x77, 0, 0x79, x25, x0, x24\n\t"
#if INT8
static inline __attribute__((always_inline)) void mac22(const uint8_t* a, uint32_t stride,
                                                        const uint8_t* b, uint32_t steps) {
    const uint8_t *t, *u;
    asm volatile(
        "1:\n\t"
        "vle8.v v8, (%0)\n\taddi %4, %0, 16\n\tvle8.v v24, (%4)\n\tadd %3, %0, %5\n\t"
        "vle8.v v16, (%1)\n\taddi %4, %1, 16\n\tvle8.v v0, (%4)\n\t"
        "vle8.v v10, (%3)\n\taddi %4, %3, 16\n\tvle8.v v26, (%4)\n\tadd %3, %3, %5\n\t"
        "addi %4, %1, 32\n\tvle8.v v18, (%4)\n\taddi %4, %1, 48\n\tvle8.v v2, (%4)\n\t"
        "vle8.v v12, (%3)\n\taddi %4, %3, 16\n\tvle8.v v28, (%4)\n\tadd %3, %3, %5\n\t"
        "addi %4, %1, 64\n\tvle8.v v20, (%4)\n\taddi %4, %1, 80\n\tvle8.v v4, (%4)\n\t"
        "vle8.v v14, (%3)\n\taddi %4, %3, 16\n\tvle8.v v30, (%4)\n\tadd %0, %3, %5\n\t"
        "addi %4, %1, 96\n\tvle8.v v22, (%4)\n\taddi %4, %1, 112\n\tvle8.v v6, (%4)\n\t"
        VTMM4
        "addi %1, %1, 128\n\taddi %2, %2, -1\n\tbnez %2, 1b"
        : "+r"(a), "+r"(b), "+r"(steps), "=&r"(t), "=&r"(u)
        : "r"(stride)
        : "v0", "v2", "v4", "v6", "v8", "v10", "v12", "v14", "v16", "v18", "v20", "v22",
          "v24", "v26", "v28", "v30", "memory");
}
#else
// Packed row pair in `hi` -> low nibble (sign-extended) to `lo`, high to `hi`.
#define UNPK(lo, hi) "vsll.vi " lo ", " hi ", 4\n\tvsra.vi " lo ", " lo ", 4\n\tvsra.vi " hi ", " hi ", 4\n\t"
static inline __attribute__((always_inline)) void mac22(const uint8_t* a, uint32_t stride,
                                                        const uint8_t* b, uint32_t steps) {
    const uint8_t *t, *u;
    asm volatile(
        "1:\n\t"
        "vle8.v v10, (%0)\n\tadd %3, %0, %5\n\t"
        "vle8.v v18, (%1)\n\taddi %4, %1, 16\n\t"
        "vle8.v v14, (%3)\n\t" UNPK("v8", "v10")
        "vle8.v v2, (%4)\n\taddi %4, %1, 32\n\t" UNPK("v16", "v18")
        "vle8.v v22, (%4)\n\taddi %4, %0, 16\n\t" UNPK("v12", "v14")
        "vle8.v v26, (%4)\n\taddi %4, %1, 48\n\t" UNPK("v0", "v2")
        "vle8.v v6, (%4)\n\taddi %4, %3, 16\n\t" UNPK("v20", "v22")
        "vle8.v v30, (%4)\n\t" UNPK("v24", "v26") UNPK("v4", "v6")
        ".insn r 0x77, 0, 0x79, x1, x16, x8\n\t"
        UNPK("v28", "v30")
        ".insn r 0x77, 0, 0x79, x9, x0, x8\n\t"
        ".insn r 0x77, 0, 0x79, x17, x16, x24\n\t"
        ".insn r 0x77, 0, 0x79, x25, x0, x24\n\t"
        "add %0, %3, %5\n\taddi %1, %1, 64\n\taddi %2, %2, -1\n\tbnez %2, 1b"
        : "+r"(a), "+r"(b), "+r"(steps), "=&r"(t), "=&r"(u)
        : "r"(stride)
        : "v0", "v2", "v4", "v6", "v8", "v10", "v12", "v14", "v16", "v18", "v20", "v22",
          "v24", "v26", "v28", "v30", "memory");
}
#endif

// ---- epilogue: 32 px x 32 output channels (block ob) -> obuf -----------------
E2E_INLINE vint32m4_t requant(vint32m4_t v, uint32_t oc) {
    v = __riscv_vadd_vx_i32m4(v, params[oc][0], 16);
    v = __riscv_vsra_vx_i32m4(__riscv_vmulh_vx_i32m4(v, params[oc][1], 16), kShift, 16);
#if RELU
    v = __riscv_vmax_vx_i32m4(v, 0, 16);
#endif
    return v;
}
static void epilogue(uint32_t ob, uint8_t* out, uint32_t p0) {
    for (uint32_t i = 0; i < 2; ++i) {
        // Junk pixels (x >= Wo on the flat grid, or past the end) become 0:
        // they are the next layer's zero padding.
        const vuint32m4_t pix =
            __riscv_vadd_vx_u32m4(__riscv_vid_v_u32m4(16), p0 + i * 16, 16);
        const vbool8_t junk = __riscv_vmor_mm_b8(
            __riscv_vmsgeu_vx_u32m4_b8(__riscv_vremu_vx_u32m4(pix, kWg, 16), kWo, 16),
            __riscv_vmsgeu_vx_u32m4_b8(pix, kP, 16), 16);
        for (uint32_t g = 0; g < 2; ++g) {
            const uint32_t tile = acc_tile(i * 2 + g) | kColumnSlice;
            const uint32_t oc0 = ob * 32 + g * 16;
#if INT8
            for (uint32_t j = 0; j < 16; ++j) {
                vint8m1_t q = to_i8(requant(vtmv(tile | j), oc0 + j));
                q = __riscv_vmerge_vxm_i8m1(q, 0, junk, 16);
                __riscv_vse8_v_i8m1(reinterpret_cast<int8_t*>(out) + (oc0 + j) * 32 + i * 16, q, 16);
            }
#else
            for (uint32_t j = 0; j < 16; j += 2) {
                vuint8m1_t q = pack_i4(requant(vtmv(tile | j), oc0 + j),
                                       requant(vtmv(tile | (j + 1)), oc0 + j + 1));
                q = __riscv_vmerge_vxm_u8m1(q, 0, junk, 16);
                __riscv_vse8_v_u8m1(out + (oc0 + j) / 2 * 32 + i * 16, q, 16);
            }
#endif
        }
    }
}

// Copy the halo (tail of the previous window) to the head of the new one.
static void copy_halo(const uint8_t* from, uint8_t* to) {
    for (uint32_t j = 0; j < kInJobs; ++j) {
        const uint8_t* src = from + j * kWin + kG;
        uint8_t* dst = to + j * kWin;
        for (uint32_t n = 0; n < kHalo;) {
            const size_t vl = __riscv_vsetvl_e8m4(kHalo - n);
            __riscv_vse8_v_u8m4(dst + n, __riscv_vle8_v_u8m4(src + n, vl), vl);
            n += vl;
        }
    }
}

static void issue_weights(uint32_t seq) { // chunk seq of the whole run
    if (kResident) return;
    dma(kWeights + (seq % kChunksPerGroup) * kChunk, addr(wbuf + (seq & 1) * kChunk), kChunk);
}

int main() {
    matrix_on();
    for (uint32_t t = 0; t < kTaps; ++t) {
        const uint32_t ky = t / KW, kx = t % KW;
        tap_off[t] = kS == 1 ? ky * kWp + kx
                             : ((ky & 1) * 2 + (kx & 1)) * kWin + (ky >> 1) * kWg + (kx >> 1);
    }
    roi_begin();
    dma(kParamsFlash, addr(params), kParams);
    if (kResident) {
        dma(kWeights, addr(wbuf), kWeightBytes);
    } else {
        issue_weights(0);
    }
    in_strip = 0; in_dst = addr(sbuf[0]); in_j = 0; in_end = kInJobs;
    issue_in(kInJobs);
    dma_wait();
    mark();

    const uint32_t total_chunks = kGroups * kChunksPerGroup;
    uint32_t seq = 0;
    for (uint32_t s = 0; s < kStrips; ++s) {
        uint8_t* strip = sbuf[kStripBufs == 2 ? (s & 1) : 0];
        if (s > 0) {
            if (kStripBufs == 2) {
                copy_halo(sbuf[(s - 1) & 1], strip);
            } else { // single buffer: slide the halo down, then load in place
                copy_halo(strip, strip);
                in_strip = s; in_dst = addr(strip); in_j = 0; in_end = kInJobs;
                issue_in(kInJobs);
                dma_wait();
            }
        }
        if (kStripBufs == 2 && s + 1 < kStrips) { // prefetch the next strip
            in_strip = s + 1; in_dst = addr(sbuf[(s + 1) & 1]); in_j = 0; in_end = kInJobs;
        }
        for (uint32_t gi = 0; gi < kNG; ++gi) {
            const uint32_t g = s * kNG + gi;
            if (g >= kGroups) break;
            uint8_t* out = obuf[kObufs == 2 ? (g & 1) : 0];
            for (uint32_t ob = 0; ob < kOcBlocks; ++ob) {
                for (uint32_t c = 0; c < kChunksPerBlock; ++c, ++seq) {
                    dma_wait();
                    if (seq + 1 < total_chunks) issue_weights(seq + 1);
                    issue_in(kInPerChunk);
                    issue_out(kOutPerChunk);
                    if (c == 0) vtzero(true);
                    const uint32_t t = c / kKChunks, kc = c % kKChunks;
                    const uint8_t* b = kResident
                                           ? wbuf + ((ob * kChunksPerBlock + c) * kChunk)
                                           : wbuf + (seq & 1) * kChunk;
                    const uint8_t* a = strip + gi * 32 + tap_off[t] +
                                       kc * (kKc / kChPerPlane) * kPlaneStride;
                    config_int8(16);
                    mac22(a, kPlaneStride, b, kKc / 4);
                    if (c == kChunksPerBlock - 1) {
                        if (kObufs == 1 && ob == 0 && out_j < out_end) {
                            issue_out(out_end);
                            dma_wait();
                        }
                        epilogue(ob, out, g * 32);
                    }
                }
            }
            issue_out(out_end); // the previous group's results, if any left
            out_group = g; out_src = addr(out); out_j = 0; out_end = kOutPlanes;
        }
        issue_in(in_end);
    }
    issue_out(out_end);
    dma_wait();
    roi_end();
    return 0;
}
