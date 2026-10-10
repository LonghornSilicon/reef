// vme_helpers.h — readable wrappers for Coral NPU's VME (Zvt) matrix engine.
//
// The toolchain does not know VME instructions yet, so each wrapper emits the
// raw encoding. Encodings and the configuration sequence follow Coral's own
// tests (Apache 2.0, google-coral/coralnpu):
//   tests/cocotb/vme_test/vme_test_utils.h
//   tests/cocotb/vme_test/vme_matmul_test_program.cc
//
// Build with -DVME_EMULATE to replace every wrapper with a plain C model of
// what the instruction does. That lets you test tiling and indexing logic on
// any machine. It does NOT test the encodings; only Coral's RTL model can.
//
// Rules for the real (non-emulated) build:
//  * VME instructions read FIXED vector registers: the matmul reads A slices
//    from v8/v10/v12/v14 and B slices from v16/v18/v20/v22; the tile-row move
//    writes v4..v7. So loads + matmul live in ONE asm block (matmul_step) and
//    the row move + store live in ONE asm block (store_tile_row).
//  * The config wrappers overwrite vtype/vl. Don't mix them with RVV
//    intrinsics in the same function unless you re-run vsetvl afterwards.
//  * Geometry below assumes VLEN = 128 (Coral's value): 16x16 tiles.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define VME_INLINE static inline __attribute__((always_inline))

namespace vme {

constexpr uint32_t kTileDim = 16; // C tile is 16 x 16 int32 (VLEN / 8)
constexpr uint32_t kKPerStep =
    4; // int8 matmul consumes 4 k-slices per instruction

// ---------------------------------------------------------------------------
// Encoding helpers (compile-time constants)
// ---------------------------------------------------------------------------
namespace enc {
constexpr uint32_t kOpV = 0x57;  // OP-V:  config, tile moves, vtzero
constexpr uint32_t kOpVE = 0x77; // OP-VE: matrix arithmetic

// [31:26 funct6] [25 vm=1] [24:20 vs2] [19:15 vs1/rs1] [14:12 funct3] [11:7 rd]
// [6:0 opcode]
constexpr uint32_t word(uint32_t funct6, uint32_t vs2, uint32_t rs1,
                        uint32_t funct3, uint32_t rd, uint32_t opcode) {
    return (funct6 << 26) | (1u << 25) | (vs2 << 20) | (rs1 << 15) |
           (funct3 << 12) | (rd << 7) | opcode;
}
// vtmmu.tvv (unsigned A) / vtmms.tvv (signed A): tile += A^T * B, A in v8.., B
// in v16..
constexpr uint32_t matmul_int(uint32_t tile, bool signed_a) {
    return word(0x3C, 8, 16, 0, (tile << 1) | (signed_a ? 1u : 0u), kOpVE);
}
// vtzero: tile = 0
constexpr uint32_t zero(uint32_t tile) {
    return word(0x10, 30, 0, 6, tile << 1, kOpV);
}
// vtmv.v.t v4, a0: tile slice selected by a0 -> v4..v7
constexpr uint32_t kMoveTileRowToV4 = word(0x10, 31, 10, 6, 4, kOpV);

// mtype CSR value: tm[23:10] | tk[7:5] | mtwiden[1:0]
constexpr uint32_t mtype(uint32_t tm, uint32_t tk, uint32_t mtwiden) {
    return ((tm & 0x3FFF) << 10) | ((tk & 0x7) << 5) | (mtwiden & 0x3);
}
// vtype values: vma | vta | vsew | vlmul
constexpr uint32_t kVtypeE8M1 =
    0xC0; // 8-bit elements, 1 register  (int8 operands)
constexpr uint32_t kVtypeE32M4 =
    0xD2; // 32-bit elements, 4 registers (one int32 tile row)

// Tile slice selector: tile[30:27] | pattern[26:24] (0 = row) | index[23:0]
constexpr uint32_t tile_row(uint32_t tile, uint32_t row) {
    return (tile << 27) | row;
}
} // namespace enc

#ifndef VME_EMULATE
// ===========================================================================
// Real instructions
// ===========================================================================

// msetmtype: mtype <- mtype_value, vtype <- vtype_value, vl <- 0.
VME_INLINE void set_mtype(uint32_t mtype_value, uint32_t vtype_value) {
    asm volatile(".insn r 0b1010111, 0b111, 0b1000001, x0, %0, %1"
                 :
                 : "r"(mtype_value), "r"(vtype_value));
}
// msettn: tile columns (and vl) <- min(n, hardware max). Returns the value set.
VME_INLINE uint32_t set_tn(uint32_t n) {
    uint32_t out = 0;
    asm volatile(".insn r 0b1010111, 0b111, 0b1000010, %0, %1, x0"
                 : "=r"(out)
                 : "r"(n));
    return out;
}
// msettm: tile rows <- min(n, hardware max).
VME_INLINE uint32_t set_tm(uint32_t n) {
    uint32_t out = 0;
    asm volatile(".insn r 0b1010111, 0b111, 0b1000010, %0, %1, x1"
                 : "=r"(out)
                 : "r"(n));
    return out;
}
// msettk: k-slices per matmul <- min(n, 4 for int8). Use for a k % 4 tail.
VME_INLINE uint32_t set_tk(uint32_t n) {
    uint32_t out = 0;
    asm volatile(".insn r 0b1010111, 0b111, 0b1000010, %0, %1, x2"
                 : "=r"(out)
                 : "r"(n));
    return out;
}

// Turn on matrix state (mstatus.MS). Matrix instructions trap as illegal
// (mcause 2) while it is Off. Coral's startup code enables FP and vector state
// but not matrix state; call this once if you see that trap.
VME_INLINE void enable_matrix_state() {
    asm volatile("csrs mstatus, %0" : : "r"(1u << 29));
}

// Mode for whole-tile operations on int32 data (vtzero, row moves).
VME_INLINE void config_tile_moves() {
    set_mtype(enc::mtype(kTileDim, 1, /*mtwiden=*/1), enc::kVtypeE32M4);
    set_tn(kTileDim); // msetmtype zeroed vl; this sets vl = 16
}

// Mode for int8 x int8 -> int32 matmul, 4 k-slices per instruction.
// Also leaves vl = 16 for the 16-byte operand loads.
VME_INLINE void config_int8_matmul() {
    set_mtype(enc::mtype(kTileDim, kKPerStep, /*mtwiden=*/3), enc::kVtypeE8M1);
    set_tm(kTileDim);
    set_tn(kTileDim);
}

// vtzero: tile <- 0. Call after config_tile_moves().
template <uint32_t TILE> VME_INLINE void zero_tile() {
    asm volatile(".word %0" : : "i"(enc::zero(TILE)) : "memory");
}

// One accumulate step, k-slices k0..k0+3:
//   tile[r][j] += sum_{t<4} A[r][k0+t] * B[k0+t][j]     for r, j in 0..15
// a_col0 = &A[row0][k0]  (column k0 of A; rows are a_stride bytes apart)
// b_row0 = &B[k0][col0]  (row k0 of B; rows are b_stride bytes apart)
// SIGNED_A picks vtmms (int8 A) over vtmmu (uint8 A). B is always uint8.
template <uint32_t TILE, bool SIGNED_A>
VME_INLINE void matmul_step(const uint8_t* a_col0, size_t a_stride,
                            const uint8_t* b_row0, size_t b_stride) {
    asm volatile(
        "vlse8.v v8,  (%[a0]), %[as]\n" // A column k0+0 (16 rows, strided)
        "vlse8.v v10, (%[a1]), %[as]\n" // A column k0+1
        "vlse8.v v12, (%[a2]), %[as]\n" // A column k0+2
        "vlse8.v v14, (%[a3]), %[as]\n" // A column k0+3
        "vle8.v  v16, (%[b0])\n"        // B row k0+0 (16 contiguous bytes)
        "vle8.v  v18, (%[b1])\n"        // B row k0+1
        "vle8.v  v20, (%[b2])\n"        // B row k0+2
        "vle8.v  v22, (%[b3])\n"        // B row k0+3
        ".word %[mm]\n"                 // vtmmu/vtmms tile, v8, v16
        :
        : [a0] "r"(a_col0), [a1] "r"(a_col0 + 1), [a2] "r"(a_col0 + 2),
          [a3] "r"(a_col0 + 3), [as] "r"(a_stride), [b0] "r"(b_row0),
          [b1] "r"(b_row0 + b_stride), [b2] "r"(b_row0 + (2 * b_stride)),
          [b3] "r"(b_row0 + (3 * b_stride)),
          [mm] "i"(enc::matmul_int(TILE, SIGNED_A))
        : "v8", "v10", "v12", "v14", "v16", "v18", "v20", "v22", "memory");
}

// Copy tile row `row` (16 int32s) to memory. Call after config_tile_moves().
template <uint32_t TILE>
// NOLINTNEXTLINE(readability-non-const-parameter): the asm stores through dst
VME_INLINE void store_tile_row(uint32_t row, int32_t* dst) {
    register uint32_t sel asm("a0") =
        enc::tile_row(TILE, row);         // encoding reads a0
    asm volatile(".word %[mv]\n"          // vtmv.v.t v4, a0
                 "vse32.v v4, (%[dst])\n" // store v4..v7
                 :
                 : [mv] "i"(enc::kMoveTileRowToV4), "r"(sel), [dst] "r"(dst)
                 : "v4", "v5", "v6", "v7", "memory");
}

#else
// ===========================================================================
// Emulation: plain C model of each instruction, for testing kernel logic.
// ===========================================================================
// Matrix tile state. Mutable and global because it models hardware registers.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
inline int32_t g_tiles[16][kTileDim][kTileDim];

VME_INLINE void enable_matrix_state() {}
VME_INLINE void config_tile_moves() {}
VME_INLINE void config_int8_matmul() {}
VME_INLINE uint32_t set_tk(uint32_t n) { return n < kKPerStep ? n : kKPerStep; }

template <uint32_t TILE> VME_INLINE void zero_tile() {
    for (auto& row : g_tiles[TILE]) {
        for (auto& x : row) {
            x = 0;
        }
    }
}

template <uint32_t TILE, bool SIGNED_A>
VME_INLINE void matmul_step(const uint8_t* a_col0, size_t a_stride,
                            const uint8_t* b_row0, size_t b_stride) {
    for (uint32_t t = 0; t < kKPerStep; t++) {
        for (uint32_t r = 0; r < kTileDim; r++) {
            const uint8_t raw = a_col0[t + (r * a_stride)];
            const int32_t a =
                SIGNED_A ? static_cast<int32_t>(static_cast<int8_t>(raw))
                         : static_cast<int32_t>(raw);
            for (uint32_t j = 0; j < kTileDim; j++) {
                g_tiles[TILE][r][j] +=
                    a * static_cast<int32_t>(b_row0[(t * b_stride) + j]);
            }
        }
    }
}

template <uint32_t TILE>
VME_INLINE void store_tile_row(uint32_t row, int32_t* dst) {
    for (uint32_t j = 0; j < kTileDim; j++) {
        dst[j] = g_tiles[TILE][row][j];
    }
}
#endif

} // namespace vme
