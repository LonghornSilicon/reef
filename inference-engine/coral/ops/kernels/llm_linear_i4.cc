// int4 (W4A4) version of coral/kernels/linear.cc: out = act[K][tokens] x
// W[K x N], weights streamed from flash by double-buffered DMA, Zvt GEMM,
// requant epilogue. Same tiling as the firmware: N/64 column groups, each K
// rows of 64 weights (+ 64 int32 bias), DMA'd in chunks of 32 rows; per step
// 4 A rows (tokens in lanes) x 4x4 B rows -> 4 vtmm into 4 accumulators.
//
//   -DK=512 -DN=2048     shape (K % 32 == 0, N % 64 == 0)
//   -DNMAT=3             back-to-back matrices of the same shape (qkv = 3)
//   -DTOKENS=1|8         tile rows tm (decode = 1, prefill block = 8)
//   -DINT4=1|0           1: weights + activations packed int4 (this kernel)
//                        0: the same kernel with int8 weights/acts (baseline)
//   -DFUSED=1            int4 only: unpack weights straight into the B operand
//                        registers inside the MAC loop (no DTCM round trip)
//   -DNATIVE4=1          int4 only: PROXY for hypothetical native-int4 Zvt
//                        (k=8 per vtmm, same operand bytes, same array time):
//                        the int8 mac4 runs directly on the packed bytes, K/2
//                        steps, no unpack. Real MACs = 2x reported useful MACs.
//   -DMODE=0|1|2         epilogue: plain | GeLU | += residual
//
// int4 layout (row-pair nibble packing, ops.hpp):
//   weights: a 32-row chunk of a column group is 1 KB packed; byte j of the
//            chunk holds element j of rows 0..15 (low nibble) and of rows
//            16..31 (high nibble), so one unpack_i4_rows(chunk, buf, 1024)
//            yields the int8 chunk in the firmware's [row][64] layout.
//   activations [feature][8 tokens]: each 32-feature block is 128 B packed,
//            features 0..15 low nibble, 16..31 high nibble. The epilogue's
//            acc pairs (0,1) and (2,3) are exactly such feature pairs.
// Weights are symmetric signed int4 (no zero point), so the firmware's
// 128*sum(act) correction is not needed in either variant here.

#include "ops.hpp"

using namespace ops;

#ifndef K
#define K 512
#endif
#ifndef N
#define N 2048
#endif
#ifndef NMAT
#define NMAT 1
#endif
#ifndef TOKENS
#define TOKENS 1
#endif
#ifndef INT4
#define INT4 1
#endif
#ifndef NATIVE4
#define NATIVE4 0
#endif
#if NATIVE4 && !INT4
#error "NATIVE4 needs INT4=1"
#endif
#ifndef FUSED
#define FUSED 0
#endif
#ifndef MODE
#define MODE 0
#endif
static_assert(K % 32 == 0 && N % 64 == 0, "shape");

constexpr uint32_t kT = 8;          // activation row stride (block of 8 tokens)
constexpr uint32_t kRows = 32;      // weight rows per chunk
constexpr uint32_t kChunks = K / kRows;
constexpr uint32_t kGroups = N / 64;
constexpr uint32_t kChunkBytes = kRows * 64 / (INT4 ? 2 : 1); // 1 KB | 2 KB
constexpr uint32_t kGroupBytes = K * 64 / (INT4 ? 2 : 1) + 256;
constexpr uint32_t kMatBytes = kGroups * kGroupBytes;
constexpr uint32_t kActBytes = K * kT / (INT4 ? 2 : 1);
constexpr uint32_t kOutBytes = N * kT / (INT4 ? 2 : 1);
constexpr uint32_t kShift = INT4 ? 12 : 8; // placeholder requant scale

static uint8_t act_in[kActBytes + 16];     // input activations (packed if int4)
static uint8_t out[kOutBytes + 16];        // output (packed if int4)
static uint8_t wdma[2][kChunkBytes + 256]; // DMA double buffer (+ bias trailer)
#if INT4 && !NATIVE4
static int8_t act8[K * kT + 16];           // activations unpacked once per call
#endif
#if INT4 && !NATIVE4 && !FUSED
static int8_t wbuf[kRows * 64];            // one unpacked int8 weight chunk
#endif
#if !INT4
static int8_t gelu_table[256];
#endif

static void issue(uint32_t mat, uint32_t index) {
    const uint32_t group = index / kChunks, chunk = index % kChunks;
    dma(kFlash + mat * kMatBytes + group * kGroupBytes + chunk * kChunkBytes,
        addr(wdma[index & 1]), kChunkBytes + (chunk == kChunks - 1 ? 256 : 0));
}

#if INT4 && FUSED
// Fused int4 MAC: 8 steps (32 rows) of one 1 KB packed chunk. Packed row p
// (64 B, v24..v27 as m4) holds weight rows p (low nibbles) and p+16 (high).
// For each 4-row set: load 4 packed rows once (4 m4 loads), A rows r..r+3,
// 4 vtmm with B = sign-extended low nibbles, then A rows r+16..r+19 and 4 vtmm
// with B = high nibbles. B goes register to register: no DTCM round trip.
E2E_INLINE void mac4_i4(const int8_t* a, const uint8_t* packed) {
    const int8_t* a_hi = a + 16 * kT;
    uint32_t sets = 4;
    asm volatile(
        "1:\n\t"
        "vsetvli x0, %4, e8, m4, ta, ma\n\t"
        "vle8.v v24, (%1)\n\taddi %1, %1, 64\n\t"
        "vle8.v v28, (%1)\n\taddi %1, %1, 64\n\t"
        "vle8.v v0, (%1)\n\taddi %1, %1, 64\n\t"
        "vle8.v v4, (%1)\n\taddi %1, %1, 64\n\t"
        "vsetivli x0, 16, e8, m1, ta, ma\n\t"
        ".insn r 0x57, 7, 0x42, x0, %5, x0\n\t"   // msetmtype int8, tm rows
        ".insn r 0x57, 7, 0x44, x0, %6, x0\n\t"   // msettn 16
        "vle8.v v8, (%0)\n\taddi %0, %0, 8\n\t"
        "vle8.v v10, (%0)\n\taddi %0, %0, 8\n\t"
        "vle8.v v12, (%0)\n\taddi %0, %0, 8\n\t"
        "vle8.v v14, (%0)\n\taddi %0, %0, 8\n\t"
        // group g: slice g of each packed row (v24+g, v28+g, v0+g, v4+g)
#define LOW(s0, s1, s2, s3, rd)                                                \
        "vsll.vi v16, " s0 ", 4\n\tvsra.vi v16, v16, 4\n\t"                    \
        "vsll.vi v18, " s1 ", 4\n\tvsra.vi v18, v18, 4\n\t"                    \
        "vsll.vi v20, " s2 ", 4\n\tvsra.vi v20, v20, 4\n\t"                    \
        "vsll.vi v22, " s3 ", 4\n\tvsra.vi v22, v22, 4\n\t"                    \
        ".insn r 0x77, 0, 0x79, " #rd ", x16, x8\n\t"
#define HIGH(s0, s1, s2, s3, rd)                                               \
        "vsra.vi v16, " s0 ", 4\n\tvsra.vi v18, " s1 ", 4\n\t"                 \
        "vsra.vi v20, " s2 ", 4\n\tvsra.vi v22, " s3 ", 4\n\t"                 \
        ".insn r 0x77, 0, 0x79, " #rd ", x16, x8\n\t"
        LOW("v24", "v28", "v0", "v4", x1)
        LOW("v25", "v29", "v1", "v5", x9)
        LOW("v26", "v30", "v2", "v6", x17)
        LOW("v27", "v31", "v3", "v7", x25)
        "vle8.v v8, (%3)\n\taddi %3, %3, 8\n\t"
        "vle8.v v10, (%3)\n\taddi %3, %3, 8\n\t"
        "vle8.v v12, (%3)\n\taddi %3, %3, 8\n\t"
        "vle8.v v14, (%3)\n\taddi %3, %3, 8\n\t"
        HIGH("v24", "v28", "v0", "v4", x1)
        HIGH("v25", "v29", "v1", "v5", x9)
        HIGH("v26", "v30", "v2", "v6", x17)
        HIGH("v27", "v31", "v3", "v7", x25)
        "addi %2, %2, -1\n\t"
        "bnez %2, 1b"
#undef LOW
#undef HIGH
        : "+r"(a), "+r"(packed), "+r"(sets), "+r"(a_hi)
        : "r"(64), "r"((TOKENS << 10) | (4U << 5) | 3U), "r"(16)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10", "v12",
          "v14", "v16", "v18", "v20", "v22", "v24", "v25", "v26", "v27", "v28",
          "v29", "v30", "v31", "vl", "vtype", "memory");
}
#endif

E2E_INLINE vint32m4_t requant(vint32m4_t acc, int32_t bias) {
    vint32m4_t v = __riscv_vsra_vx_i32m4(__riscv_vadd_vx_i32m4(acc, bias, 16), kShift, 16);
#if MODE == 1 // GeLU approximation used by the firmware table: x>0 ? x : x/8
    v = __riscv_vmax_vv_i32m4(v, __riscv_vsra_vx_i32m4(v, 3, 16), 16);
#endif
    return v;
}

// Column group g's 4 accumulators -> 64 output features (rows of kT bytes).
static void epilogue(uint32_t g, const int32_t* bias) {
#if INT4
    for (uint32_t pair = 0; pair < 2; ++pair) {
        uint8_t* blk = out + g * 256 + pair * 128; // 32-feature packed block
        for (uint32_t j = 0; j < 16; ++j) {
            vint32m4_t lo = requant(vtmv(acc_tile(2 * pair) | kColumnSlice | j),
                                    bias[32 * pair + j]);
            vint32m4_t hi = requant(vtmv(acc_tile(2 * pair + 1) | kColumnSlice | j),
                                    bias[32 * pair + 16 + j]);
#if MODE == 2 // residual (packed int4) += result
            const vint8m1_t r = __riscv_vreinterpret_v_u8m1_i8m1(
                __riscv_vle8_v_u8m1(blk + j * kT, 16));
            lo = __riscv_vadd_vv_i32m4(lo, __riscv_vsext_vf4_i32m4(
                __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(r, 4, 16), 4, 16), 16), 16);
            hi = __riscv_vadd_vv_i32m4(hi, __riscv_vsext_vf4_i32m4(
                __riscv_vsra_vx_i8m1(r, 4, 16), 16), 16);
#endif
            __riscv_vse8_v_u8m1(blk + j * kT, pack_i4(lo, hi), TOKENS);
        }
    }
#else
    for (uint32_t acc = 0; acc < 4; ++acc) {
        for (uint32_t j = 0; j < 16; ++j) {
            int8_t* row = reinterpret_cast<int8_t*>(out) + (g * 64 + acc * 16 + j) * kT;
            vint32m4_t v = requant(vtmv(acc_tile(acc) | kColumnSlice | j), bias[acc * 16 + j]);
#if MODE == 2
            v = __riscv_vadd_vv_i32m4(v, load_i8(row), 16);
#endif
            vint8m1_t r = to_i8(v);
#if MODE == 1
            r = __riscv_vluxei8_v_i8m1(gelu_table, __riscv_vreinterpret_v_i8m1_u8m1(r), 16);
#endif
            __riscv_vse8_v_i8m1(row, r, TOKENS);
        }
    }
#endif
}

static void linear(uint32_t mat) {
    constexpr uint32_t total = kGroups * kChunks;
    issue(mat, 0);
#if NATIVE4
    // Native int4 proxy: packed activations used as-is (K/2 rows of kT bytes).
    const int8_t* act = reinterpret_cast<const int8_t*>(act_in);
#elif INT4
    // Activations: unpacked once per call, reused by all N/64 column groups.
    for (uint32_t b = 0; b < K / 32; ++b) {
        unpack_i4_rows(act_in + b * 128, act8 + b * 256, 128);
    }
    const int8_t* act = act8;
#else
    const int8_t* act = reinterpret_cast<const int8_t*>(act_in);
#endif
    for (uint32_t i = 0; i < total; ++i) {
        dma_wait();
        if (i + 1 < total) issue(mat, i + 1);
        const uint32_t chunk = i % kChunks;
        const uint8_t* buf = wdma[i & 1];
        if (chunk == 0) vtzero(true);
#if NATIVE4
        // One packed 1 KB chunk = 16 rows of 64 B, each byte 2 k-values: 4 steps.
        config_int8(TOKENS);
        mac4(act + chunk * (kRows / 2) * kT, kT, buf, kRows / 8);
#elif INT4 && FUSED
        mac4_i4(act + chunk * kRows * kT, buf);
#else
#if INT4
        // Weights are used once per decode token: this unpack is not amortized.
        unpack_i4_rows(buf, wbuf, kChunkBytes);
        const void* w = wbuf;
#else
        const void* w = buf;
#endif
        config_int8(TOKENS);
        mac4(act + chunk * kRows * kT, kT, w, kRows / 4);
#endif
        if (chunk == kChunks - 1) {
            epilogue(i / kChunks, reinterpret_cast<const int32_t*>(buf + kChunkBytes));
        }
    }
}

int main() {
    matrix_on();
#if !INT4
    for (uint32_t i = 0; i < 256; ++i) {
        const auto v = static_cast<int8_t>(i);
        gelu_table[i] = v > 0 ? v : static_cast<int8_t>(v >> 3);
    }
#endif
    roi_begin();
    for (uint32_t m = 0; m < NMAT; ++m) {
        linear(m);
        mark();
    }
    roi_end();
    return 0;
}
