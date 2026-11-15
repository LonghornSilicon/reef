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
//    channels, 4 vtmm = 4096 MACs): two 32-byte packed A loads and one
//    64-byte packed B load (mac4 int8: 20 x 16-byte loads), unpacked in
//    registers (fused unpack, see mac22).
//  * Loop order: strip (NG pixel groups, rolling halo) > group of 32 px >
//    32-channel output block > tap/K-chunk. Weights are streamed from flash
//    per pixel group (double-buffered chunks) unless the whole layer fits the
//    DTCM, in which case they are loaded once (resident).
//  * The input strip is double-buffered (if it fits); only the new G bytes
//    per plane are DMAed, the halo is copied from the previous strip's tail.
//  * Weight flash layout: [32-ch block][tap][K-chunk] chunks, then per output
//    channel {int32 bias, int32 Q32 multiplier}. PSRAM: input plane-phases
//    at kIn, output planes at kOut (next layer's padded grid).
//
// Shapes: -DCIN -DCOUT -DH -DW -DKH -DKW -DSTRIDE -DPAD -DRELU, -DGROUPS=n
// runs only the first n pixel groups (a slice). -DINT8=1: int8 everything
// (same tiling, no packing). -DABL: profiling ablations (see below).
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
#ifndef ABL
#define ABL 0 // ablation for profiling: 1 = no epilogue, 2 = no MAC loop
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
constexpr uint32_t kAvail = 32768 - 1024 - 128 - 192; // stack, guard, small statics
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
// One struct so the order is fixed: the inner loop reads one k-step past
// its operands (sbuf -> wbuf, wbuf -> obuf), all inside the struct.
static struct {
    uint8_t sbuf[kStripBufs][kStripBuf];
    uint8_t wbuf[kWbuf];
    uint8_t obuf[kObufs][kObuf1];
    int32_t params[kCout][2]; // bias, multiplier (Q31)
    uint8_t guard[128];
} dt __attribute__((aligned(16)));
static_assert(2 * kPlaneStride * kChPerPlane <= kWbuf, "over-read guard");
// Build-time configuration, readable from the ELF (.rodata, "conf" symbol).
extern "C" __attribute__((used)) const uint32_t conv2d_conf[] = {
    kNG, kResident, kStripBufs, kObufs, kWin, kChunk, kGroups, kStrips, kWeightBytes, sizeof(dt)};
#define sbuf dt.sbuf
#define wbuf dt.wbuf
#define obuf dt.obuf
#define params dt.params
static uint16_t tap_off[kTaps];

// ---- DMA job streams (spread over the chunk iterations) --------------------
// A stream is a run of equal-length copies with fixed source/destination
// steps (one per plane): the next strip's input planes, or the previous
// group's output planes. A few are issued per chunk so they never queue up
// in front of the weight chunk the next iteration waits for.
struct Stream {
    uint32_t src, dst, len, left, sstep, dstep;
};
E2E_INLINE void pump(Stream& st, uint32_t n) {
    for (; n && st.left; --n, --st.left) {
        dma(st.src, st.dst, st.len);
        st.src += st.sstep;
        st.dst += st.dstep;
    }
}
constexpr uint32_t kInJobs = kInPlanes * kPhases;
constexpr uint32_t kChunksPerStrip = kNG * kChunksPerGroup;
constexpr uint32_t kInPerChunk = cdiv(kInJobs, kChunksPerStrip);
constexpr uint32_t kOutPerChunk = cdiv(kOutPlanes, kChunksPerGroup);
// Strip s: window [s*G, s*G + G + halo) of every plane-phase; after the
// first strip only the new G bytes (the halo is copied in DTCM).
E2E_INLINE Stream in_stream(uint32_t s, uint8_t* buf) {
    return s == 0 ? Stream{kIn, addr(buf), kG + kHalo, kInJobs, kInPhase, kWin}
                  : Stream{kIn + s * kG + kHalo, addr(buf) + kHalo, kG, kInJobs, kInPhase, kWin};
}
E2E_INLINE Stream out_stream(uint32_t g, const uint8_t* buf) {
    const uint32_t p0 = g * 32, len = kP - p0 < 32 ? kP - p0 : 32;
    return Stream{addr(buf), kOut + kOutBase + p0, len, kOutPlanes, 32, kOutPlane};
}

// ---- inner loop: 2 pixel tiles x 2 channel groups, `steps` k-steps ----------
// The core's 8-entry ROB holds every instruction until it completes (a vtmm
// ~20 cycles), so the loop is written for few, wide instructions:
//  * 32-byte (LMUL 2) A loads: tile 0 | tile 1 of one channel pair land in a
//    register pair, so one load feeds the A bases vN (tile 0) and vN+1
//    (tile 1); one 64-byte (LMUL 4) B load per k-step feeds vM (group 0) and
//    vM+1 (group 1). Zvt accepts int8 operand bases v0/1/8/9/16/17/24/25.
//  * Unpack in place, 2 ops per register pair: lo = b << 4, hi = b & 0xF0,
//    i.e. 16 x the int4 value, exact in int8. The accumulator is 256 x the
//    true sum (|sum| < 2^27 for K <= 4608); the requantizer drops the 256.
//    Within a vtmm the k-slots are (c0, c2, c1, c3); A and B agree.
//  * vl = 32/64 needs vsetvli, which clears mtype, so every vtmm group is
//    preceded by msetmtype + msettn (tm 16, tk 4, int8).
//  * Two register sets (E: A v8-15, B v16-23; O: A v24-31, B v0-7),
//    software-pipelined: the loads and unpack of one set run under the
//    other set's vtmms. 2 k-steps per iteration; steps must be even. The
//    loop reads one k-step past the end (guard bytes after the buffers).
// mt0 = t0 x g0, mt4 = t0 x g1, mt8 = t1 x g0, mt12 = t1 x g1.
#if INT8
#define MAC22_BODY \
        "vsetvli x0, %[n64], e8, m4, ta, ma\n\t" \
        "vle8.v v16, (%[b])\n\t" \
        "addi %[b], %[b], 64\n\t" \
        "vle8.v v20, (%[b])\n\t" \
        "addi %[b], %[b], 64\n\t" \
        "vsetvli x0, %[n32], e8, m2, ta, ma\n\t" \
        "vle8.v v8, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "vle8.v v10, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "vle8.v v12, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "vle8.v v14, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "1:\n\t" \
        "vsetvli x0, %[n64], e8, m4, ta, ma\n\t" \
        "vle8.v v0, (%[b])\n\t" \
        "addi %[b], %[b], 64\n\t" \
        "vle8.v v4, (%[b])\n\t" \
        "addi %[b], %[b], 64\n\t" \
        ".insn r 0x57, 7, 0x42, x0, %[mt], x0\n\t" \
        ".insn r 0x57, 7, 0x44, x0, %[n16], x0\n\t" \
        ".insn r 0x77, 0, 0x79, x1, x16, x8\n\t" \
        ".insn r 0x77, 0, 0x79, x9, x17, x8\n\t" \
        ".insn r 0x77, 0, 0x79, x17, x16, x9\n\t" \
        ".insn r 0x77, 0, 0x79, x25, x17, x9\n\t" \
        "vsetvli x0, %[n32], e8, m2, ta, ma\n\t" \
        "vle8.v v24, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "vle8.v v26, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "vle8.v v28, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "vle8.v v30, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "vsetvli x0, %[n64], e8, m4, ta, ma\n\t" \
        "vle8.v v16, (%[b])\n\t" \
        "addi %[b], %[b], 64\n\t" \
        "vle8.v v20, (%[b])\n\t" \
        "addi %[b], %[b], 64\n\t" \
        ".insn r 0x57, 7, 0x42, x0, %[mt], x0\n\t" \
        ".insn r 0x57, 7, 0x44, x0, %[n16], x0\n\t" \
        ".insn r 0x77, 0, 0x79, x1, x0, x24\n\t" \
        ".insn r 0x77, 0, 0x79, x9, x1, x24\n\t" \
        ".insn r 0x77, 0, 0x79, x17, x0, x25\n\t" \
        ".insn r 0x77, 0, 0x79, x25, x1, x25\n\t" \
        "vsetvli x0, %[n32], e8, m2, ta, ma\n\t" \
        "vle8.v v8, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "vle8.v v10, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "vle8.v v12, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "vle8.v v14, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "addi %[n], %[n], -1\n\tbnez %[n], 1b"
#else
#define MAC22_BODY \
        "vsetvli x0, %[n64], e8, m4, ta, ma\n\t" \
        "vle8.v v20, (%[b])\n\t" \
        "addi %[b], %[b], 64\n\t" \
        "vsetvli x0, %[n32], e8, m2, ta, ma\n\t" \
        "vle8.v v12, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "vle8.v v14, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "vsll.vi v8, v12, 4\n\t" \
        "vand.vi v12, v12, -16\n\t" \
        "vsll.vi v10, v14, 4\n\t" \
        "vand.vi v14, v14, -16\n\t" \
        "vsll.vi v16, v20, 4\n\t" \
        "vand.vi v20, v20, -16\n\t" \
        "vsll.vi v18, v22, 4\n\t" \
        "vand.vi v22, v22, -16\n\t" \
        "1:\n\t" \
        "vsetvli x0, %[n64], e8, m4, ta, ma\n\t" \
        "vle8.v v4, (%[b])\n\t" \
        "addi %[b], %[b], 64\n\t" \
        ".insn r 0x57, 7, 0x42, x0, %[mt], x0\n\t" \
        ".insn r 0x57, 7, 0x44, x0, %[n16], x0\n\t" \
        ".insn r 0x77, 0, 0x79, x1, x16, x8\n\t" \
        ".insn r 0x77, 0, 0x79, x9, x17, x8\n\t" \
        ".insn r 0x77, 0, 0x79, x17, x16, x9\n\t" \
        ".insn r 0x77, 0, 0x79, x25, x17, x9\n\t" \
        "vsetvli x0, %[n32], e8, m2, ta, ma\n\t" \
        "vle8.v v28, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "vle8.v v30, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "vsll.vi v24, v28, 4\n\t" \
        "vand.vi v28, v28, -16\n\t" \
        "vsll.vi v26, v30, 4\n\t" \
        "vand.vi v30, v30, -16\n\t" \
        "vsll.vi v0, v4, 4\n\t" \
        "vand.vi v4, v4, -16\n\t" \
        "vsll.vi v2, v6, 4\n\t" \
        "vand.vi v6, v6, -16\n\t" \
        "vsetvli x0, %[n64], e8, m4, ta, ma\n\t" \
        "vle8.v v20, (%[b])\n\t" \
        "addi %[b], %[b], 64\n\t" \
        ".insn r 0x57, 7, 0x42, x0, %[mt], x0\n\t" \
        ".insn r 0x57, 7, 0x44, x0, %[n16], x0\n\t" \
        ".insn r 0x77, 0, 0x79, x1, x0, x24\n\t" \
        ".insn r 0x77, 0, 0x79, x9, x1, x24\n\t" \
        ".insn r 0x77, 0, 0x79, x17, x0, x25\n\t" \
        ".insn r 0x77, 0, 0x79, x25, x1, x25\n\t" \
        "vsetvli x0, %[n32], e8, m2, ta, ma\n\t" \
        "vle8.v v12, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "vle8.v v14, (%[a])\n\t" \
        "add %[a], %[a], %[s]\n\t" \
        "vsll.vi v8, v12, 4\n\t" \
        "vand.vi v12, v12, -16\n\t" \
        "vsll.vi v10, v14, 4\n\t" \
        "vand.vi v14, v14, -16\n\t" \
        "vsll.vi v16, v20, 4\n\t" \
        "vand.vi v20, v20, -16\n\t" \
        "vsll.vi v18, v22, 4\n\t" \
        "vand.vi v22, v22, -16\n\t" \
        "addi %[n], %[n], -1\n\tbnez %[n], 1b"
#endif
static inline __attribute__((always_inline)) void mac22(const uint8_t* a, uint32_t stride,
                                                        const uint8_t* b, uint32_t steps) {
    uint32_t iters = steps / 2;
    asm volatile(
        MAC22_BODY
        : [a] "+&r"(a), [b] "+&r"(b), [n] "+&r"(iters)
        : [s] "r"(stride), [mt] "r"((16U << 10) | (4U << 5) | 3U), [n16] "r"(16U),
          [n32] "r"(32U), [n64] "r"(64U)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "vtype", "memory");
}

// ---- epilogue: 16 px (tile i) x 32 output channels (block ob) -> obuf -------
// Hand-written so the int32 tile mode is armed once per batch of 8 (int8: 4)
// columns: vtmv one output channel x 16 px, bias + Q32 multiply (the shift
// is folded into the multiplier) + clamp (ReLU: [0, 7]) at e32, pack channel
// pairs into nibbles, then narrow e32 -> e16 -> e8 once per batch (any
// vsetvli clears mtype). Moves are issued ahead of the math that uses them.
// Junk pixels (x >= Wo on the flat grid, or past the end) are zeroed: they
// are the next layer's zero padding.
static void epilogue_tile(uint32_t sel, const int32_t* p, uint8_t* o, uint32_t junk) {
    constexpr int32_t lo = RELU ? 0 : (INT8 ? -128 : -8), hi = INT8 ? 127 : 7;
    for (uint32_t b = 0; b < (INT8 ? 8 : 4); ++b) {
        if (b == (INT8 ? 4 : 2)) sel += acc_tile(1) - 16; // second accumulator (group 1)
        asm volatile(
#if INT8
            "vsetivli x0, 1, e32, m1, ta, ma\n\t"
            "vmv.s.x v0, %[m]\n\t"
            ".insn r 0x57, 7, 0x42, x0, %[mt32], %[vt32]\n\t"
            ".insn r 0x57, 7, 0x44, x0, %[n16], x0\n\t"
            ".insn r 0x57, 6, 0x21, x16, %[sel], x31\n\t"
            "addi %[sel], %[sel], 1\n\t"
            ".insn r 0x57, 6, 0x21, x20, %[sel], x31\n\t"
            "addi %[sel], %[sel], 1\n\t"
            "lw t0, 0(%[p])\n\t"
            "lw t1, 4(%[p])\n\t"
            "vadd.vx v16, v16, t0\n\t"
            "vmulh.vx v16, v16, t1\n\t"
            "vmax.vx v16, v16, %[lo]\n\t"
            "vmin.vx v16, v16, %[hi]\n\t"
            ".insn r 0x57, 6, 0x21, x24, %[sel], x31\n\t"
            "addi %[sel], %[sel], 1\n\t"
            "lw t2, 8(%[p])\n\t"
            "lw t3, 12(%[p])\n\t"
            "vadd.vx v20, v20, t2\n\t"
            "vmulh.vx v20, v20, t3\n\t"
            "vmax.vx v20, v20, %[lo]\n\t"
            "vmin.vx v20, v20, %[hi]\n\t"
            ".insn r 0x57, 6, 0x21, x28, %[sel], x31\n\t"
            "addi %[sel], %[sel], 1\n\t"
            "lw t0, 16(%[p])\n\t"
            "lw t1, 20(%[p])\n\t"
            "vadd.vx v24, v24, t0\n\t"
            "vmulh.vx v24, v24, t1\n\t"
            "vmax.vx v24, v24, %[lo]\n\t"
            "vmin.vx v24, v24, %[hi]\n\t"
            "lw t2, 24(%[p])\n\t"
            "lw t3, 28(%[p])\n\t"
            "vadd.vx v28, v28, t2\n\t"
            "vmulh.vx v28, v28, t3\n\t"
            "vmax.vx v28, v28, %[lo]\n\t"
            "vmin.vx v28, v28, %[hi]\n\t"
            "addi %[p], %[p], 32\n\t"
            "vsetivli x0, 16, e16, m2, ta, ma\n\t"
            "vnsrl.wi v16, v16, 0\n\t"
            "vnsrl.wi v20, v20, 0\n\t"
            "vnsrl.wi v24, v24, 0\n\t"
            "vnsrl.wi v28, v28, 0\n\t"
            "vsetivli x0, 16, e8, m1, ta, ma\n\t"
            "vnsrl.wi v16, v16, 0\n\t"
            "vnsrl.wi v20, v20, 0\n\t"
            "vnsrl.wi v24, v24, 0\n\t"
            "vnsrl.wi v28, v28, 0\n\t"
            "vmerge.vim v16, v16, 0, v0\n\t"
            "vse8.v v16, (%[o])\n\t"
            "addi %[o], %[o], 32\n\t"
            "vmerge.vim v20, v20, 0, v0\n\t"
            "vse8.v v20, (%[o])\n\t"
            "addi %[o], %[o], 32\n\t"
            "vmerge.vim v24, v24, 0, v0\n\t"
            "vse8.v v24, (%[o])\n\t"
            "addi %[o], %[o], 32\n\t"
            "vmerge.vim v28, v28, 0, v0\n\t"
            "vse8.v v28, (%[o])\n\t"
            "addi %[o], %[o], 32\n\t"
#else
            "vsetivli x0, 1, e32, m1, ta, ma\n\t"
            "vmv.s.x v0, %[m]\n\t"
            ".insn r 0x57, 7, 0x42, x0, %[mt32], %[vt32]\n\t"
            ".insn r 0x57, 7, 0x44, x0, %[n16], x0\n\t"
            ".insn r 0x57, 6, 0x21, x16, %[sel], x31\n\t"
            "addi %[sel], %[sel], 1\n\t"
            ".insn r 0x57, 6, 0x21, x12, %[sel], x31\n\t"
            "addi %[sel], %[sel], 1\n\t"
            ".insn r 0x57, 6, 0x21, x20, %[sel], x31\n\t"
            "addi %[sel], %[sel], 1\n\t"
            ".insn r 0x57, 6, 0x21, x8, %[sel], x31\n\t"
            "addi %[sel], %[sel], 1\n\t"
            "lw t0, 0(%[p])\n\t"
            "lw t1, 4(%[p])\n\t"
            "lw t2, 8(%[p])\n\t"
            "lw t3, 12(%[p])\n\t"
            "vadd.vx v16, v16, t0\n\t"
            "vmulh.vx v16, v16, t1\n\t"
            "vmax.vx v16, v16, %[lo]\n\t"
            "vmin.vx v16, v16, %[hi]\n\t"
            "vadd.vx v12, v12, t2\n\t"
            "vmulh.vx v12, v12, t3\n\t"
            "vmax.vx v12, v12, %[lo]\n\t"
            "vmin.vx v12, v12, %[hi]\n\t"
#if !RELU
            "vand.vi v16, v16, 15\n\t"
#endif
            "vsll.vi v12, v12, 4\n\t"
            "vor.vv v16, v16, v12\n\t"
            "lw t0, 16(%[p])\n\t"
            "lw t1, 20(%[p])\n\t"
            "lw t2, 24(%[p])\n\t"
            "lw t3, 28(%[p])\n\t"
            ".insn r 0x57, 6, 0x21, x24, %[sel], x31\n\t"
            "addi %[sel], %[sel], 1\n\t"
            ".insn r 0x57, 6, 0x21, x4, %[sel], x31\n\t"
            "addi %[sel], %[sel], 1\n\t"
            "vadd.vx v20, v20, t0\n\t"
            "vmulh.vx v20, v20, t1\n\t"
            "vmax.vx v20, v20, %[lo]\n\t"
            "vmin.vx v20, v20, %[hi]\n\t"
            "vadd.vx v8, v8, t2\n\t"
            "vmulh.vx v8, v8, t3\n\t"
            "vmax.vx v8, v8, %[lo]\n\t"
            "vmin.vx v8, v8, %[hi]\n\t"
#if !RELU
            "vand.vi v20, v20, 15\n\t"
#endif
            "vsll.vi v8, v8, 4\n\t"
            "vor.vv v20, v20, v8\n\t"
            "lw t0, 32(%[p])\n\t"
            "lw t1, 36(%[p])\n\t"
            "lw t2, 40(%[p])\n\t"
            "lw t3, 44(%[p])\n\t"
            ".insn r 0x57, 6, 0x21, x28, %[sel], x31\n\t"
            "addi %[sel], %[sel], 1\n\t"
            ".insn r 0x57, 6, 0x21, x12, %[sel], x31\n\t"
            "addi %[sel], %[sel], 1\n\t"
            "vadd.vx v24, v24, t0\n\t"
            "vmulh.vx v24, v24, t1\n\t"
            "vmax.vx v24, v24, %[lo]\n\t"
            "vmin.vx v24, v24, %[hi]\n\t"
            "vadd.vx v4, v4, t2\n\t"
            "vmulh.vx v4, v4, t3\n\t"
            "vmax.vx v4, v4, %[lo]\n\t"
            "vmin.vx v4, v4, %[hi]\n\t"
#if !RELU
            "vand.vi v24, v24, 15\n\t"
#endif
            "vsll.vi v4, v4, 4\n\t"
            "vor.vv v24, v24, v4\n\t"
            "lw t0, 48(%[p])\n\t"
            "lw t1, 52(%[p])\n\t"
            "lw t2, 56(%[p])\n\t"
            "lw t3, 60(%[p])\n\t"
            "vadd.vx v28, v28, t0\n\t"
            "vmulh.vx v28, v28, t1\n\t"
            "vmax.vx v28, v28, %[lo]\n\t"
            "vmin.vx v28, v28, %[hi]\n\t"
            "vadd.vx v12, v12, t2\n\t"
            "vmulh.vx v12, v12, t3\n\t"
            "vmax.vx v12, v12, %[lo]\n\t"
            "vmin.vx v12, v12, %[hi]\n\t"
#if !RELU
            "vand.vi v28, v28, 15\n\t"
#endif
            "vsll.vi v12, v12, 4\n\t"
            "vor.vv v28, v28, v12\n\t"
            "addi %[p], %[p], 64\n\t"
            "vsetivli x0, 16, e16, m2, ta, ma\n\t"
            "vnsrl.wi v16, v16, 0\n\t"
            "vnsrl.wi v20, v20, 0\n\t"
            "vnsrl.wi v24, v24, 0\n\t"
            "vnsrl.wi v28, v28, 0\n\t"
            "vsetivli x0, 16, e8, m1, ta, ma\n\t"
            "vnsrl.wi v16, v16, 0\n\t"
            "vnsrl.wi v20, v20, 0\n\t"
            "vnsrl.wi v24, v24, 0\n\t"
            "vnsrl.wi v28, v28, 0\n\t"
            "vmerge.vim v16, v16, 0, v0\n\t"
            "vse8.v v16, (%[o])\n\t"
            "addi %[o], %[o], 32\n\t"
            "vmerge.vim v20, v20, 0, v0\n\t"
            "vse8.v v20, (%[o])\n\t"
            "addi %[o], %[o], 32\n\t"
            "vmerge.vim v24, v24, 0, v0\n\t"
            "vse8.v v24, (%[o])\n\t"
            "addi %[o], %[o], 32\n\t"
            "vmerge.vim v28, v28, 0, v0\n\t"
            "vse8.v v28, (%[o])\n\t"
            "addi %[o], %[o], 32\n\t"
#endif
            : [sel] "+&r"(sel), [p] "+&r"(p), [o] "+&r"(o)
            : [mt32] "r"((16U << 10) | (1U << 5) | 1U), [vt32] "r"(2U << 3), [n16] "r"(16U),
              [lo] "r"(lo), [hi] "r"(hi), [m] "r"(junk)
            : "t0", "t1", "t2", "t3", "v0", "v1", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
              "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "vl", "vtype",
              "memory");
    }
}
static uint32_t junk_mask[2]; // per tile of the current group: bit = junk pixel
static void make_junk_masks(uint32_t p0) {
    uint32_t x = p0 % kWg;
    for (uint32_t i = 0; i < 32; ++i) {
        if (i % 16 == 0) junk_mask[i / 16] = 0;
        if (x >= kWo || p0 + i >= kP) junk_mask[i / 16] |= 1U << (i % 16);
        if (++x == kWg) x = 0;
    }
}
[[maybe_unused]] static void epilogue(uint32_t ob, uint8_t* out) {
    for (uint32_t i = 0; i < 2; ++i) {
        epilogue_tile(acc_tile(2 * i) | kColumnSlice, &params[ob * 32][0],
                      out + ob * 32 / kChPerPlane * 32 + i * 16, junk_mask[i]);
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

int main() {
    asm volatile("" ::"r"(conv2d_conf)); // keep the configuration record
    matrix_on();
    for (uint32_t t = 0; t < kTaps; ++t) {
        const uint32_t ky = t / KW, kx = t % KW;
        tap_off[t] = kS == 1 ? ky * kWp + kx
                             : ((ky & 1) * 2 + (kx & 1)) * kWin + (ky >> 1) * kWg + (kx >> 1);
    }
    roi_begin();
    dma(kParamsFlash, addr(params), kParams);
    // Weight chunks stream in sequence (block, tap, K-chunk), the same
    // sequence for every pixel group, double-buffered.
    uint32_t wsrc = kWeights;
    if (kResident) {
        dma(kWeights, addr(wbuf), kWeightBytes);
    } else {
        dma(wsrc, addr(wbuf), kChunk);
    }
    Stream in = in_stream(0, sbuf[0]), out{0, 0, 0, 0, 0, 0};
    pump(in, kInJobs);
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
                in = in_stream(s, strip);
                pump(in, kInJobs);
                dma_wait();
            }
        }
        if (kStripBufs == 2 && s + 1 < kStrips) in = in_stream(s + 1, sbuf[(s + 1) & 1]);
        for (uint32_t gi = 0; gi < kNG; ++gi) {
            const uint32_t g = s * kNG + gi;
            if (g >= kGroups) break;
            uint8_t* obp = obuf[kObufs == 2 ? (g & 1) : 0];
            make_junk_masks(g * 32);
            const uint8_t* resident_b = wbuf;
            for (uint32_t ob = 0; ob < kOcBlocks; ++ob) {
                for (uint32_t c = 0; c < kChunksPerBlock; ++c, ++seq) {
                    dma_wait();
                    if (!kResident && seq + 1 < total_chunks) {
                        wsrc += kChunk;
                        if (wsrc == kWeights + kWeightBytes) wsrc = kWeights;
                        dma(wsrc, addr(wbuf + ((seq + 1) & 1) * kChunk), kChunk);
                    }
                    if (kInPerChunk) pump(in, kInPerChunk);
                    pump(out, kOutPerChunk);
                    if (c == 0) vtzero(true);
                    const uint32_t t = kKChunks == 1 ? c : c / kKChunks;
                    const uint32_t kc = kKChunks == 1 ? 0 : c % kKChunks;
                    const uint8_t* b = kResident ? resident_b : wbuf + (seq & 1) * kChunk;
                    resident_b += kChunk;
                    const uint8_t* a = strip + gi * 32 + tap_off[t] +
                                       kc * (kKc / kChPerPlane) * kPlaneStride;
#if ABL != 2 && ABL != 3
                    mac22(a, kPlaneStride, b, kKc / 4);
#else
                    (void)a; (void)b;
#endif
                    if (c == kChunksPerBlock - 1) {
                        if (kObufs == 1 && ob == 0 && out.left) {
                            pump(out, out.left);
                            dma_wait();
                        }
#if ABL != 1 && ABL != 3
                        epilogue(ob, obp);
#endif
                    }
                }
            }
            pump(out, out.left); // the previous group's results, if any left
            out = out_stream(g, obp);
        }
        pump(in, in.left);
    }
    pump(out, out.left);
    dma_wait();
    roi_end();
    return 0;
}
