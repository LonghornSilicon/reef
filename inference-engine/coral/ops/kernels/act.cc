// Elementwise activations (ReLU, LeakyReLU, SiLU, GeLU, Tanh, Sigmoid) on
// packed int4 feature maps. Every non-ReLU activation is a table lookup, so
// the table CONTENTS (ACT) never change the timing; only METHOD does.
//
//   -DMODE=0  stream: PSRAM -> DTCM -> apply -> PSRAM, double-buffered 6 KB
//             chunks (2 in + 2 out buffers), over a MAP_C x MAP_H x MAP_W map.
//   -DMODE=1  fused: 8 KB in + 8 KB out in DTCM, no DMA, REPS passes. The
//             per-element cost an epilogue pays (subtract METHOD=0 for the
//             load/store the epilogue already does).
//   -DLM=m1|m4  vector register group for the apply loop (m1 = 16 bytes,
//             what a conv/GEMM epilogue's pack_i4 produces; m4 = 64 bytes).
//
//   METHOD (data = packed int4, 2 per byte, unless noted):
//   0  copy: load + store only (no activation): the baseline
//   1  ReLU on packed int4 with bit tricks: zero each nibble whose sign bit is set
//   2  unpack nibbles (and / srl, unsigned 0..15) -> 2x vluxei8 into two
//      16-entry tables (low-nibble table, high-nibble table pre-shifted) -> or
//   3  unpack to signed int8 (sll/sra) -> 2x vluxei8 into the 256-entry int8
//      table (the linear.cc kGelu path) -> pack (and/sll/or)
//   4  byte-indexed 256-entry table on the packed byte: ONE vluxei8, no unpack
//   5  like 2 but the 16-entry tables live in vector registers: 2x vrgather
//   6  int8 storage (1 element per byte): ReLU = vmax
//   7  int8 storage: vluxei8 into the 256-entry int8 table (linear.cc kGelu)
//
// reef-perf runs the code functionally (Spike), so indexed-load addresses come
// from the data actually in DTCM; DMA moves no data, so the buffers are filled
// with pseudo-random bytes before the ROI and keep them.
#include "ops.hpp"

using namespace ops;

#ifndef MODE
#define MODE 0
#endif
#ifndef METHOD
#define METHOD 4
#endif
#ifndef ACT
#define ACT 2 // 0 ReLU, 1 LeakyReLU(0.2), 2 SiLU, 3 GeLU, 4 Tanh, 5 Sigmoid
#endif
#ifndef LM
#define LM m4
#endif
#ifndef MAP_C
#define MAP_C 64
#endif
#ifndef MAP_H
#define MAP_H 80
#endif
#ifndef MAP_W
#define MAP_W 80
#endif
#ifndef REPS
#define REPS 128
#endif

constexpr bool kInt8 = METHOD >= 6;
constexpr uint32_t kElems = MAP_C * MAP_H * MAP_W;
constexpr uint32_t kBytes = kInt8 ? kElems : kElems / 2; // bytes in PSRAM
constexpr uint32_t kChunk = 6144; // stream chunk (2 in + 2 out buffers)
constexpr uint32_t kFused = 8192; // fused: 8 KB in + 8 KB out, no DMA

// int4 activation tables: f(q * 0.5) requantized to int4. Output scales:
// 0.5 (ReLU/Leaky/SiLU/GeLU), 1/7 (Tanh), 1/16 with zero point -8 (Sigmoid).
// Index = the nibble as unsigned 0..15 (8..15 are -8..-1).
static constexpr int8_t kAct16[6][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0},              // ReLU
    {0, 1, 2, 3, 4, 5, 6, 7, -2, -1, -1, -1, -1, -1, 0, 0},        // LeakyReLU
    {0, 1, 1, 2, 4, 5, 6, 7, 0, 0, 0, 0, 0, -1, -1, 0},            // SiLU
    {0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0},              // GeLU
    {0, 3, 5, 6, 7, 7, 7, 7, -7, -7, -7, -7, -7, -6, -5, -3},      // Tanh
    {0, 2, 4, 5, 6, 7, 7, 7, -8, -8, -7, -7, -6, -5, -4, -2},      // Sigmoid
};

struct alignas(16) Tables {
    uint8_t lo16[16];   // method 2/5: f(n) & 0xF
    uint8_t hi16[16];   // method 2/5: f(n) << 4
    uint8_t byte256[256]; // method 4: byte -> 2 activated nibbles
    int8_t int8_256[256]; // method 3/7: signed int8 -> int8 (clamped input)
};
static Tables tables;

// Out of place so the input bytes (and so the table lines touched) stay the
// same pseudo-random data on every pass; DMA moves no data in reef-perf.
#if MODE == 0
static uint8_t inb[2][kChunk] __attribute__((aligned(16)));
static uint8_t outb[2][kChunk] __attribute__((aligned(16)));
#else
static uint8_t inb[1][kFused] __attribute__((aligned(16)));
static uint8_t outb[1][kFused] __attribute__((aligned(16)));
#endif

static void init_tables() {
    const int8_t* f = kAct16[ACT];
    for (uint32_t n = 0; n < 16; ++n) {
        tables.lo16[n] = static_cast<uint8_t>(f[n] & 0xF);
        tables.hi16[n] = static_cast<uint8_t>((f[n] & 0xF) << 4);
    }
    for (uint32_t b = 0; b < 256; ++b) {
        tables.byte256[b] = static_cast<uint8_t>(tables.lo16[b & 15] | tables.hi16[b >> 4]);
        int32_t v = static_cast<int8_t>(b);
        v = v < -8 ? -8 : v > 7 ? 7 : v; // int4 range
        tables.int8_256[b] = f[v & 15];
    }
}

// Pseudo-random data so indexed loads touch realistic table lines.
static void fill(uint8_t* p, uint32_t n) {
    uint32_t s = 0x12345678u;
    for (uint32_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        p[i] = static_cast<uint8_t>(s >> 24);
    }
}

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define U8 CAT(CAT(vuint8, LM), _t)
#define I8 CAT(CAT(vint8, LM), _t)
#define VSETVL CAT(__riscv_vsetvl_e8, LM)
#define VLE CAT(__riscv_vle8_v_u8, LM)
#define VSE CAT(__riscv_vse8_v_u8, LM)
#define VAND CAT(__riscv_vand_vx_u8, LM)
#define VSRL CAT(__riscv_vsrl_vx_u8, LM)
#define VSLL CAT(__riscv_vsll_vx_u8, LM)
#define VOR CAT(__riscv_vor_vv_u8, LM)
#define VXOR CAT(__riscv_vxor_vx_u8, LM)
#define VMUL CAT(__riscv_vmul_vx_u8, LM)
#define VANDV CAT(__riscv_vand_vv_u8, LM)
#define VLUX CAT(__riscv_vluxei8_v_u8, LM)
#define VRGATHER CAT(__riscv_vrgather_vv_u8, LM)
#define TO_I8 CAT(__riscv_vreinterpret_v_u8, CAT(LM, CAT(_i8, LM)))
#define TO_U8 CAT(__riscv_vreinterpret_v_i8, CAT(LM, CAT(_u8, LM)))
#define VSLL_I CAT(__riscv_vsll_vx_i8, LM)
#define VSRA_I CAT(__riscv_vsra_vx_i8, LM)
#define VMAX_I CAT(__riscv_vmax_vx_i8, LM)

// Apply the activation to n bytes (n % 16 == 0): in -> out.
static void apply(const uint8_t* p, uint8_t* out, uint32_t n) {
#if METHOD == 5
    const size_t vlmax = VSETVL(64);
    U8 tlo = VLE(tables.lo16, 16); // 16 entries; indices are < 16
    U8 thi = VLE(tables.hi16, 16);
    (void)vlmax;
#endif
    for (uint32_t i = 0; i < n;) {
        const size_t vl = VSETVL(n - i);
        U8 b = VLE(p + i, vl);
#if METHOD == 0
        U8 r = b;
#elif METHOD == 1
        // t = sign bits moved to bit 0 / bit 4; t * 15 = nibble masks.
        U8 m = VMUL(VSRL(VAND(b, 0x88, vl), 3, vl), 15, vl);
        U8 r = VANDV(b, VXOR(m, 0xFF, vl), vl);
#elif METHOD == 2
        U8 lo = VLUX(tables.lo16, VAND(b, 0x0F, vl), vl);
        U8 hi = VLUX(tables.hi16, VSRL(b, 4, vl), vl);
        U8 r = VOR(lo, hi, vl);
#elif METHOD == 3
        I8 s = TO_I8(b);
        U8 lo = VLUX(reinterpret_cast<const uint8_t*>(tables.int8_256),
                     TO_U8(VSRA_I(VSLL_I(s, 4, vl), 4, vl)), vl);
        U8 hi = VLUX(reinterpret_cast<const uint8_t*>(tables.int8_256),
                     TO_U8(VSRA_I(s, 4, vl)), vl);
        U8 r = VOR(VAND(lo, 0x0F, vl), VSLL(hi, 4, vl), vl);
#elif METHOD == 4
        U8 r = VLUX(tables.byte256, b, vl);
#elif METHOD == 5
        U8 lo = VRGATHER(tlo, VAND(b, 0x0F, vl), vl);
        U8 hi = VRGATHER(thi, VSRL(b, 4, vl), vl);
        U8 r = VOR(lo, hi, vl);
#elif METHOD == 6
        U8 r = TO_U8(VMAX_I(TO_I8(b), 0, vl));
#elif METHOD == 7
        U8 r = VLUX(reinterpret_cast<const uint8_t*>(tables.int8_256), b, vl);
#endif
        VSE(out + i, r, vl);
        i += vl;
    }
}

int main() {
    init_tables();
#if MODE == 0
    fill(inb[0], 2 * kChunk);
    constexpr uint32_t kChunks = (kBytes + kChunk - 1) / kChunk;
    const uint32_t src = kRam, dst = kRam + (4u << 20);
    roi_begin();
    dma(src, addr(inb[0]), kChunk);
    for (uint32_t c = 0; c < kChunks; ++c) {
        dma_wait(); // chunk c is in; chunk c-1's write-back (outb[c-1&1]) is out
        const uint32_t len = c + 1 < kChunks ? kChunk : kBytes - c * kChunk;
        if (c + 1 < kChunks) {
            const uint32_t next = c + 2 < kChunks ? kChunk : kBytes - (c + 1) * kChunk;
            dma(src + (c + 1) * kChunk, addr(inb[(c + 1) & 1]), next);
        }
        apply(inb[c & 1], outb[c & 1], len);
        dma(addr(outb[c & 1]), dst + c * kChunk, len);
    }
    dma_wait();
    roi_end();
#else
    fill(inb[0], kFused);
    roi_begin();
    for (uint32_t r = 0; r < REPS; ++r) {
        apply(inb[0], outb[0], kFused);
    }
    roi_end();
#endif
    return 0;
}
