// ConvTranspose2d k4 s2 p1 (DCGAN generator layers 2..5), W4A4 on Zvt.
//
// Sub-pixel decomposition: output (2m+ry, 2n+rx) = sum over the 2x2 taps of
// phase (ry, rx) of in[m+dy][n+dx] . w[ky][kx], with
//   ry = 0: ky=1 -> dy=0, ky=3 -> dy=-1      ry = 1: ky=0 -> dy=+1, ky=2 -> dy=0
// (same for x). Each phase is a dense 2x2 conv: an implicit GEMM with
// K = 4 taps x CIN, no zero multiplies.
//
// Zvt orientation: A (tm = 16 lanes) = 16 output channels (weights), B (16
// lanes per accumulator) = 16 output positions; 4 accumulators = 64 positions.
// One pass = 16 och x 64 positions x full K, accumulators never spill.
//
// Activation layout (PSRAM and DTCM): per channel a plane with row pitch
// W+1 (one zero column per row) and a zero row above and below, plus one
// leading zero byte: pixel (y, x) at 1 + (y+1)*pitch + x. Position p =
// m*pitch + n (n == W is a discarded pad position). Tap (dy, dx) of position
// p reads plane[1 + p + (1+dy)*pitch + dx]: every B row is 16 contiguous bytes,
// the borders read the zero pad. int4: channel pairs (2c, 2c+1) share a byte
// plane (low/high nibble, the row-pair convention of ops.hpp).
//
// Dataflow (pixel-group stationary):
//   for each group of <= 64 positions:
//     DMA the group's activation span for all CIN channels (packed) from PSRAM
//     and unpack it once, in place, to int8 [CIN][S] in DTCM;
//     for ry, for och block (16), for rx:  (phase = ry, rx)
//       stream the phase/och weight block (4*CIN x 16) from flash in KC-row
//       chunks, double-buffered; unpack each chunk once, reuse it across the
//       64 positions (4 accumulators);
//       epilogue: vtmv rows, *mult + bias >> shift, ReLU (Tanh LUT on the
//       last layer), pack int4 channel pairs, interleave rx into a staging row
//       (stride-2 store); after rx=1, DMA the finished output rows to PSRAM in
//       the next layer's padded layout.
//   -DCIN -DCOUT -DHIN   shape (HIN = input H = W)
//   -DINT8=1             same kernel, int8 storage (no pack/unpack)
//   -DTANH=1             Tanh LUT epilogue (last layer) instead of ReLU
#include "ops.hpp"

using namespace ops;

#ifndef CIN
#define CIN 128
#endif
#ifndef COUT
#define COUT 64
#endif
#ifndef HIN
#define HIN 16
#endif
#ifndef INT8
#define INT8 0
#endif
#ifndef TANH
#define TANH 0
#endif
#ifndef FUSED
#define FUSED 1 // int4 A rows unpacked straight into v8..v14 inside the MAC loop
#endif

constexpr uint32_t kW = HIN, kH = HIN;
constexpr uint32_t kPitch = kW + 1;
constexpr uint32_t kNpos = kH * kPitch;                    // positions per phase
constexpr uint32_t kNpos16 = (kNpos + 15) / 16 * 16;
constexpr uint32_t kNaccMax = kNpos16 / 16 < 4 ? kNpos16 / 16 : 4;
constexpr uint32_t kS = 16 * kNaccMax + 2 * kPitch + 2;    // channel span (DTCM)
constexpr uint32_t kLin = kNpos16 + 2 * kPitch + 2;        // input plane length
constexpr uint32_t kWo = 2 * kW, kPitchO = kWo + 1;
constexpr uint32_t kLout = (2 * kH * kPitchO + 15) / 16 * 16 + 2 * kPitchO + 2;
constexpr uint32_t kNob = (COUT + 15) / 16;                // 16-och blocks
constexpr uint32_t kKC = CIN < 128 ? CIN : 128;            // weight chunk rows
constexpr uint32_t kChunksPerBlock = 4 * CIN / kKC;
constexpr uint32_t kChunksPerGroup = 4 * kNob * kChunksPerBlock;
constexpr uint32_t kShift = 16;
#if INT8
constexpr uint32_t kChunkBytes = kKC * 16;
constexpr uint32_t kRowsOut = 16;                          // staging rows / ob
#else
constexpr uint32_t kChunkBytes = kKC * 8;
constexpr uint32_t kRowsOut = 8;                           // channel pairs / ob
#endif
constexpr uint32_t kBlockBytes = kChunksPerBlock * kChunkBytes;
constexpr uint32_t kWeightBytes = 4 * kNob * kBlockBytes;
constexpr uint32_t kStg = 2 * 16 * kNaccMax;               // staging row bytes
constexpr uint32_t kInBase = kRam;
constexpr uint32_t kOutBase = kRam + (4u << 20);
static_assert(CIN % kKC == 0 && kKC % 4 == 0 && CIN % 2 == 0, "shape");

static int8_t act[CIN * kS] __attribute__((aligned(16)));
#if INT8
static int8_t wbuf[2][kChunkBytes] __attribute__((aligned(16)));
static int8_t stage[16 * kStg] __attribute__((aligned(16)));
#else
static uint8_t wpk[2][kChunkBytes] __attribute__((aligned(16)));
#if !FUSED
static int8_t w8[kKC * 16] __attribute__((aligned(16)));
#endif
static uint8_t stage[8 * kStg] __attribute__((aligned(16)));
#endif
static int32_t params[kNob * 16 * 2]; // [och] mult, then [och] bias
#if TANH
static int8_t tanh_lut[256];
#endif

// ---- MAC: steps x (4 A rows of 16 och, stride 16; per accumulator g: 4 B
// rows of 16 positions at b + 16 g, channel stride bs; vtmm into mt4g) ------
#define CT_B(g, rd)                                                            \
    "addi %[t], %[b], " #g "*16\n\tvle8.v v16, (%[t])\n\tadd %[t], %[t], %[bs]\n\t" \
    "vle8.v v18, (%[t])\n\tadd %[t], %[t], %[bs]\n\tvle8.v v20, (%[t])\n\t"      \
    "add %[t], %[t], %[bs]\n\tvle8.v v22, (%[t])\n\t"                           \
    ".insn r 0x77, 0, 0x79, " #rd ", x16, x8\n\t"
#if FUSED && !INT8
// Packed k-quad (32 B): bytes 0-15 = rows k0 (low nibble) / k1 (high),
// bytes 16-31 = rows k2 / k3. Two loads + six shifts give v8, v10, v12, v14.
#define CT_A                                                                   \
    "vle8.v v24, (%[a])\n\taddi %[a], %[a], 16\n\t"                            \
    "vle8.v v26, (%[a])\n\taddi %[a], %[a], 16\n\t"                            \
    "vsll.vi v8, v24, 4\n\tvsra.vi v8, v8, 4\n\tvsra.vi v10, v24, 4\n\t"        \
    "vsll.vi v12, v26, 4\n\tvsra.vi v12, v12, 4\n\tvsra.vi v14, v26, 4\n\t"
#else
#define CT_A                                                                   \
    "vle8.v v8, (%[a])\n\taddi %[a], %[a], 16\n\t"                             \
    "vle8.v v10, (%[a])\n\taddi %[a], %[a], 16\n\t"                            \
    "vle8.v v12, (%[a])\n\taddi %[a], %[a], 16\n\t"                            \
    "vle8.v v14, (%[a])\n\taddi %[a], %[a], 16\n\t"
#endif
#define CT_MAC(name, BLOCKS)                                                   \
    static void name(const int8_t* a, const int8_t* b, uint32_t bs, uint32_t n) { \
        const int8_t* t;                                                       \
        asm volatile("1:\n\t" CT_A BLOCKS                                     \
                     "add %[b], %[b], %[bs4]\n\t"                              \
                     "addi %[n], %[n], -1\n\tbnez %[n], 1b"                    \
                     : [a] "+r"(a), [b] "+r"(b), [n] "+r"(n), [t] "=&r"(t)     \
                     : [bs] "r"(bs), [bs4] "r"(4 * bs)                         \
                     : "v8", "v10", "v12", "v14", "v16", "v18", "v20", "v22",  \
                       "v24", "v26", "memory");                                \
    }
CT_MAC(mac_1, CT_B(0, x1))
CT_MAC(mac_2, CT_B(0, x1) CT_B(1, x9))
CT_MAC(mac_3, CT_B(0, x1) CT_B(1, x9) CT_B(2, x17))
CT_MAC(mac_4, CT_B(0, x1) CT_B(1, x9) CT_B(2, x17) CT_B(3, x25))

static void mac_n(uint32_t nacc, const int8_t* a, const int8_t* b, uint32_t n) {
    config_int8(16);
    switch (nacc) {
    case 1: mac_1(a, b, kS, n); break;
    case 2: mac_2(a, b, kS, n); break;
    case 3: mac_3(a, b, kS, n); break;
    default: mac_4(a, b, kS, n); break;
    }
}

// ---- weight stream: flat chunk index within a group -> (ry, ob, rx, c) ------
struct Chunk {
    uint32_t ry, ob, rx, c;
};
static Chunk decode(uint32_t i) {
    Chunk k;
    k.c = i % kChunksPerBlock;
    i /= kChunksPerBlock;
    k.rx = i % 2;
    i /= 2;
    k.ob = i % kNob;
    k.ry = i / kNob;
    return k;
}
static void issue_chunk(uint32_t i) {
    const Chunk k = decode(i % kChunksPerGroup);
    const uint32_t block = (k.ry * 2 + k.rx) * kNob + k.ob;
#if INT8
    void* dst = wbuf[i & 1];
#else
    void* dst = wpk[i & 1];
#endif
    dma(kFlash + block * kBlockBytes + k.c * kChunkBytes, addr(dst), kChunkBytes);
}

static inline int32_t tap_d(uint32_t r, uint32_t t) { // r = phase bit, t = tap
    return r == 0 ? (t == 0 ? 0 : -1) : (t == 0 ? 1 : 0);
}

static inline vint32m4_t requant(vint32m4_t v, uint32_t och) {
    v = __riscv_vmul_vx_i32m4(v, params[och], 16);
    v = __riscv_vsra_vx_i32m4(__riscv_vadd_vx_i32m4(v, params[kNob * 16 + och], 16),
                              kShift, 16);
#if TANH
    const vint8m1_t idx = to_i8(v);
    const vint8m1_t y = __riscv_vluxei8_v_i8m1(
        tanh_lut, __riscv_vreinterpret_v_i8m1_u8m1(idx), 16);
    return __riscv_vsext_vf4_i32m4(y, 16);
#else
    return __riscv_vmax_vx_i32m4(v, 0, 16);
#endif
}

static void epilogue(uint32_t nacc, uint32_t ob, uint32_t rx) {
    const uint32_t valid = COUT - ob * 16 < 16 ? COUT - ob * 16 : 16;
    for (uint32_t g = 0; g < nacc; ++g) {
#if INT8
        for (uint32_t r = 0; r < valid; ++r) {
            const vint8m1_t y = to_i8(requant(vtmv(acc_tile(g) | r), ob * 16 + r));
            __riscv_vsse8_v_i8m1(stage + r * kStg + 32 * g + rx, 2, y, 16);
        }
#else
        for (uint32_t r = 0; r < valid; r += 2) {
            const vint32m4_t y0 = requant(vtmv(acc_tile(g) | r), ob * 16 + r);
            const vint32m4_t y1 = requant(vtmv(acc_tile(g) | (r + 1)), ob * 16 + r + 1);
            __riscv_vsse8_v_u8m1(stage + (r / 2) * kStg + 32 * g + rx, 2,
                                 pack_i4(y0, y1), 16);
        }
#endif
    }
}

// Finished (ry, ob): DMA each staging row's output-row segments to PSRAM.
static void write_out(uint32_t ry, uint32_t ob, uint32_t p0, uint32_t p1) {
    const uint32_t valid = COUT - ob * 16 < 16 ? COUT - ob * 16 : 16;
#if INT8
    const uint32_t rows = valid;
#else
    const uint32_t rows = (valid + 1) / 2;
#endif
    for (uint32_t cp = 0; cp < rows; ++cp) {
        for (uint32_t m = p0 / kPitch; m * kPitch < p1; ++m) {
            const uint32_t lo = (p0 > m * kPitch ? p0 : m * kPitch) - m * kPitch;
            const uint32_t hi_p = p1 < m * kPitch + kW ? p1 : m * kPitch + kW;
            if (hi_p <= m * kPitch + lo) continue;
            const uint32_t hi = hi_p - m * kPitch;
            dma(addr(stage + cp * kStg + 2 * (m * kPitch + lo - p0)),
                kOutBase + (ob * kRowsOut + cp) * kLout + 1 + (2 * m + ry + 1) * kPitchO +
                    2 * lo,
                2 * (hi - lo));
        }
    }
}

int main() {
    matrix_on();
    roi_begin();
    dma(kFlash + kWeightBytes, addr(params), sizeof(params));
    uint32_t flat = 0; // global chunk counter (buffer parity)
    issue_chunk(0);
    for (uint32_t p0 = 0; p0 < kNpos; p0 += 64) {
        const uint32_t npos = kNpos - p0 < 64 ? kNpos - p0 : 64;
        const uint32_t nacc = (npos + 15) / 16;
        const uint32_t span = 16 * nacc + 2 * kPitch + 2;
        // Activation span of this group, all channels; unpack once in place.
#if INT8
        for (uint32_t c = 0; c < CIN; ++c) {
            dma(kInBase + c * kLin + p0, addr(act + c * kS), span);
        }
        dma_wait();
#else
        uint8_t* packed = reinterpret_cast<uint8_t*>(act) + CIN / 2 * kS;
        for (uint32_t c = 0; c < CIN / 2; ++c) {
            dma(kInBase + c * kLin + p0, addr(packed + c * kS), span);
        }
        dma_wait();
        for (uint32_t c = 0; c < CIN / 2; ++c) {
            unpack_i4_rows(packed + c * kS, act + 2 * c * kS, kS);
        }
#endif
        mark();
        for (uint32_t i = 0; i < kChunksPerGroup; ++i, ++flat) {
            dma_wait();
            const bool more = p0 + 64 < kNpos || i + 1 < kChunksPerGroup;
            if (more) issue_chunk(flat + 1);
            const Chunk k = decode(i);
            const uint32_t row0 = k.c * kKC; // within the 4*CIN block
            const uint32_t tap = row0 / CIN, c0 = row0 % CIN;
            const int32_t dy = tap_d(k.ry, tap / 2), dx = tap_d(k.rx, tap % 2);
            const int8_t* b = act + c0 * kS + 1 + (1 + dy) * int32_t(kPitch) + dx;
#if INT8
            const int8_t* a = wbuf[flat & 1];
#elif FUSED
            const int8_t* a = reinterpret_cast<const int8_t*>(wpk[flat & 1]);
#else
            unpack_i4_rows(wpk[flat & 1], w8, kChunkBytes);
            const int8_t* a = w8;
#endif
            if (k.c == 0) vtzero(true);
            mac_n(nacc, a, b, kKC / 4);
            if (k.c == kChunksPerBlock - 1) {
                epilogue(nacc, k.ob, k.rx);
                if (k.rx == 1) write_out(k.ry, k.ob, p0, p0 + npos);
            }
        }
    }
    dma_wait();
    roi_end();
    return 0;
}
