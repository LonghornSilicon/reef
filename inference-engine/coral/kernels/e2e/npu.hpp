#pragma once

// Hardware interface for the end-to-end firmware (e2e.cc): model shape, the
// XSPI flash map, the DMA initiator and the Zvt matrix engine.
//
// Data path (RTL team, 2026-11):
//   XSPI  16 bit/cycle, ~6 ns  off-chip flash: weights (and, for now, KV cache)
//   QSPI   4 bit/cycle, ~40 ns PSRAM: KV cache (later; everything is XSPI now)
//   SPI    1 bit/cycle         host link: boots the ITCM, starts the core
// The DMA moves data between the external memories and the 32 KB DTCM
// (reef-perf memory/Dma.h). reef-perf models one external link, so the KV
// cache and the weights share it.

#include <riscv_vector.h>

#include <cstdint>

#define E2E_INLINE static inline __attribute__((always_inline))

namespace npu {

// ---- Model: TinyStories-Instruct-8M (GPT-Neo), int8 weights ---------------
#ifndef E2E_HIDDEN
#define E2E_HIDDEN 256
#endif
#ifndef E2E_LAYERS
#define E2E_LAYERS 8
#endif
#ifndef E2E_MAX_CONTEXT
#define E2E_MAX_CONTEXT 2064 // KV cache reserved per layer (positions)
#endif
#ifndef E2E_VOCAB
#define E2E_VOCAB 50257
#endif

constexpr uint32_t kHidden = E2E_HIDDEN;           // features per token
constexpr uint32_t kNumHeads = 16;                 // every TinyStories config
constexpr uint32_t kHeadDim = kHidden / kNumHeads; // 4 (1M) .. 48 (33M)
constexpr uint32_t kKvBlock = 16 * kHeadDim; // bytes: 16 positions of K or V
constexpr uint32_t kMlpDim = 4 * kHidden;    // MLP inner features
constexpr uint32_t kNumLayers = E2E_LAYERS;
constexpr uint32_t kVocab = E2E_VOCAB;
// Whole 64-column groups; the pad rows get a -inf bias.
constexpr uint32_t kVocabPadded = (kVocab + 63) / 64 * 64;
constexpr uint32_t kMaxContext = E2E_MAX_CONTEXT;
constexpr uint32_t kKvBlocks = kMaxContext / 16; // KV blocks of 16 positions
constexpr uint32_t kLocalWindow = 256;   // GPT-Neo local attention (odd layers)
constexpr uint32_t kMaxPositions = 2048; // position embeddings
static_assert(kHeadDim % 4 == 0 && kHidden % 64 == 0, "tile shapes");
// Prefill block: tokens per tile (tile rows). Activations are
// [feature][kBlockTokens].
#ifndef E2E_BLOCK_TOKENS
#define E2E_BLOCK_TOKENS 16
#endif
constexpr uint32_t kBlockTokens = E2E_BLOCK_TOKENS;

// ---- XSPI flash map (all addresses 64-byte aligned) -----------------------
// A linear layer [K x N] is stored as N/64 column groups, each K rows of 64
// uint8 weights (zero point 128) followed by 64 int32 biases. A row holds the
// B operands of the 4 vtmm (4 accumulators, 16 columns each) that share one
// A load, so every DMA is one contiguous run. This is the offline transpose
// that makes the matrix engine's operands contiguous.
constexpr uint32_t align64(uint32_t bytes) { return (bytes + 63) & ~63U; }
constexpr uint32_t linear_bytes(uint32_t in_dim, uint32_t out_dim) {
    return out_dim / 64 * (in_dim * 64 + 256);
}
constexpr uint32_t kFlashBase = 0x20000000;
constexpr uint32_t kTokenEmbed = kFlashBase; // [V][H] int8 rows
constexpr uint32_t kPosEmbed =
    align64(kTokenEmbed + kVocab * kHidden); // [2048][H]
constexpr uint32_t kNormOffset = 0;          // g1 b1 g2 b2, [H] each
constexpr uint32_t kQueryOffset = 4 * kHidden;
constexpr uint32_t kKeyOffset = kQueryOffset + linear_bytes(kHidden, kHidden);
constexpr uint32_t kValueOffset = kKeyOffset + linear_bytes(kHidden, kHidden);
constexpr uint32_t kOutOffset = kValueOffset + linear_bytes(kHidden, kHidden);
constexpr uint32_t kFc1Offset = kOutOffset + linear_bytes(kHidden, kHidden);
constexpr uint32_t kFc2Offset = kFc1Offset + linear_bytes(kHidden, kMlpDim);
constexpr uint32_t kLayerBytes =
    align64(kFc2Offset + linear_bytes(kMlpDim, kHidden));
constexpr uint32_t kLayersBase = align64(kPosEmbed + kMaxPositions * kHidden);
constexpr uint32_t kFinalNorm = kLayersBase + kNumLayers * kLayerBytes; // gf bf
constexpr uint32_t kLmHead =
    align64(kFinalNorm + 2 * kHidden); // lm_head [H x VP]
// KV cache [layer][head][block][16 D]: K blocks are K^T (D dims x 16
// positions), V blocks are V (16 positions x D dims).
constexpr uint32_t kCacheBytes = kNumLayers * kNumHeads * kKvBlocks * kKvBlock;
constexpr uint32_t kKeyCache =
    align64(kLmHead + linear_bytes(kHidden, kVocabPadded));
constexpr uint32_t kValueCache = kKeyCache + kCacheBytes;
constexpr uint32_t kFlashEnd = kValueCache + kCacheBytes;

// ---- DMA (reef-perf memory/Dma.h; register block is an assumption) -------
E2E_INLINE uint32_t addr(const void* ptr) {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr));
}
E2E_INLINE void dma(uint32_t src, uint32_t dst, uint32_t len) {
    auto* regs = reinterpret_cast<volatile uint32_t*>(0x40000000);
    regs[0] = src;
    regs[1] = dst;
    regs[2] = len;
    regs[3] = 1; // GO
}
// Returns when every started transfer is done.
E2E_INLINE void dma_wait() {
    (void)reinterpret_cast<volatile uint32_t*>(0x40000000)[4];
}

// ---- Zvt (CoralNPU matrix extension v0.4) ----------------------------------
// Encodings: reef-perf/microbenchmarks/matrix/zvt.inc. The accumulator is
// int32 tile group mt0-mt3: C[tile_rows x 16] += A^T B with A rows in v8..v14
// (tile_rows lanes) and B rows in v16..v22 (16 lanes), 4 k-steps per
// instruction.
E2E_INLINE void msetmtype(uint32_t mtype_value, uint32_t vtype_value) {
    asm volatile(".insn r 0x57, 7, 0x42, x0, %0, %1" ::"r"(mtype_value),
                 "r"(vtype_value)
                 : "vl", "vtype");
}
E2E_INLINE void msettn(uint32_t count) {
    uint32_t unused;
    asm volatile(".insn r 0x57, 7, 0x44, %0, %1, x0"
                 : "=r"(unused)
                 : "r"(count)
                 : "vl", "vtype");
}
// int32 tile slices (vtzero, vtmv): vl = 16.
E2E_INLINE void config_int32() {
    msetmtype((16U << 10) | (1U << 5) | 1U, 2U << 3);
    msettn(16);
}
// int8 MAC, tm rows (tokens), tn = 16, tk = 4. Operand loads use vl = 16.
E2E_INLINE void config_int8(uint32_t tile_rows) {
    msetmtype((tile_rows << 10) | (4U << 5) | 3U, 0);
    msettn(16);
}
// Zero mt0 or, with all, the four int32 accumulators mt0, mt4, mt8, mt12.
E2E_INLINE void vtzero(bool all = false) {
    config_int32();
    asm volatile(".insn r 0x57, 6, 0x21, x0, x0, x30" ::: "memory");
    if (all) {
        asm volatile(".insn r 0x57, 6, 0x21, x8, x0, x30\n\t"
                     ".insn r 0x57, 6, 0x21, x16, x0, x30\n\t"
                     ".insn r 0x57, 6, 0x21, x24, x0, x30" ::
                         : "memory");
    }
}
// Tile slice selector: column j (bit 24) or row j of accumulator g (mt4g).
constexpr uint32_t kColumnSlice = 1U << 24;
constexpr uint32_t acc_tile(uint32_t acc) { return (4 * acc) << 27; }
// vtmv.v.t: one int32 slice of mt0. Any vset* clears mtype (RvvFrontEnd.sv
// :205), and the compiler emits one before every intrinsic, so each read
// configures the int32 tile mode itself (msetmtype + msettn, then the
// move).
E2E_INLINE vint32m4_t vtmv(uint32_t slice) {
    vint32m4_t out;
    asm volatile(".insn r 0x57, 7, 0x42, x0, %2, %3\n\t"
                 ".insn r 0x57, 7, 0x44, x0, %4, x0\n\t"
                 ".insn r 0x57, 6, 0x21, x4, %1, x31\n\t"
                 "vmv4r.v %0, v4"
                 : "=vr"(out)
                 : "r"(slice), "r"((16U << 10) | (1U << 5) | 1U), "r"(2U << 3),
                   "r"(16U)
                 : "v4", "v5", "v6", "v7", "vl", "vtype", "memory");
    return out;
}
// steps x (4 A rows from a, stride sa, then for each accumulator g: 4 B
// rows at b + 16 g with stride 64, one vtmm into mt4g). A is int8 (vtmms).
E2E_INLINE void mac4(const void* a_rows, uint32_t a_stride, const void* b_rows,
                     uint32_t steps) {
    const void* b_ptr;
#define E2E_LOAD_B(g, rd)                                                      \
    "addi %3, %1, " #g "*16\n\tvle8.v v16, (%3)\n\taddi %3, %3, 64\n\t"        \
    "vle8.v v18, (%3)\n\taddi %3, %3, 64\n\tvle8.v v20, (%3)\n\t"              \
    "addi %3, %3, 64\n\tvle8.v v22, (%3)\n\t"                                  \
    ".insn r 0x77, 0, 0x79, " #rd ", x16, x8\n\t"
    asm volatile("1:\n\t"
                 "vle8.v v8, (%0)\n\tadd %0, %0, %4\n\t"
                 "vle8.v v10, (%0)\n\tadd %0, %0, %4\n\t"
                 "vle8.v v12, (%0)\n\tadd %0, %0, %4\n\t"
                 "vle8.v v14, (%0)\n\tadd %0, %0, %4\n\t" E2E_LOAD_B(0, x1)
                     E2E_LOAD_B(1, x9) E2E_LOAD_B(2, x17)
                         E2E_LOAD_B(3, x25) "addi %1, %1, 256\n\t"
                                            "addi %2, %2, -1\n\t"
                                            "bnez %2, 1b"
                 : "+r"(a_rows), "+r"(b_rows), "+r"(steps), "=&r"(b_ptr)
                 : "r"(a_stride)
                 : "v8", "v10", "v12", "v14", "v16", "v18", "v20", "v22",
                   "memory");
#undef E2E_LOAD_B
}
// steps x (4 A rows from a, 4 B rows from b, one vtmm into mt0), rows sa
// and sb bytes apart. kSigned: A is int8 (vtmms), else uint8 (vtmmu).
template <bool kSigned>
E2E_INLINE void mac(const void* a_rows, uint32_t a_stride, const void* b_rows,
                    uint32_t b_stride, uint32_t steps) {
    asm volatile("1:\n\t"
                 "vle8.v v8, (%0)\n\tadd %0, %0, %4\n\t"
                 "vle8.v v10, (%0)\n\tadd %0, %0, %4\n\t"
                 "vle8.v v12, (%0)\n\tadd %0, %0, %4\n\t"
                 "vle8.v v14, (%0)\n\tadd %0, %0, %4\n\t"
                 "vle8.v v16, (%1)\n\tadd %1, %1, %5\n\t"
                 "vle8.v v18, (%1)\n\tadd %1, %1, %5\n\t"
                 "vle8.v v20, (%1)\n\tadd %1, %1, %5\n\t"
                 "vle8.v v22, (%1)\n\tadd %1, %1, %5\n\t"
                 ".insn r 0x77, 0, 0x79, x%3, x16, x8\n\t"
                 "addi %2, %2, -1\n\t"
                 "bnez %2, 1b"
                 : "+r"(a_rows), "+r"(b_rows), "+r"(steps)
                 : "i"(kSigned ? 1 : 0), "r"(a_stride), "r"(b_stride)
                 : "v8", "v10", "v12", "v14", "v16", "v18", "v20", "v22",
                   "memory");
}

E2E_INLINE uint32_t mcycle() {
    uint32_t cycles;
    asm volatile("csrr %0, mcycle" : "=r"(cycles)::"memory");
    return cycles;
}

} // namespace npu
