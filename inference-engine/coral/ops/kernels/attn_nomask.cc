// No-mask (bidirectional) attention for vision: the YOLO11n / YOLO26n C2PSA
// PSABlock at P5 (20x20 = 400 tokens). Q, K, V come from the preceding 1x1
// conv (qkv) and live in PSRAM as packed int4. All N queries attend to all N
// keys, per head:  O = softmax(Q K^T) V.
//
// Per head, per query tile of 16 tokens (tm = 16 fills the tile), per key
// chunk of KC keys (KC = N for N <= 800, so N = 400 is ONE chunk):
//   1. QK^T: K streamed in groups of 64 keys (packed int4, DMA double-buffered
//      or resident in DTCM), unpacked to int8 [DK][64], mac4 -> 4 int32 tiles
//      (16 queries x 64 keys). Each tile column (= one key, 16 query lanes) is
//      read with vtmv, requantized to an int8 logit and stored to S[key][q].
//   2. Softmax over the whole chunk on int8 at e8m4 (4 keys x 16 queries per
//      instruction): row max, gap = max - s (0..255), exp LUT gather (vluxei8),
//      row sum widened to u16, P (0..127, int8) written over S in place.
//   3. PV: V streamed in groups of 32 keys (packed int4 -> int8 [32][64]),
//      mac4(P, V) accumulates O[16 q x 64 dims] in the 4 tiles over ALL keys
//      of the chunk with no rescale (exact max is known before PV).
//   4. Epilogue: O / rowsum (one vdivu per tile), packed to int4, DMA to PSRAM.
//   With more than one chunk (N = 1600, KC = 800): flash-style online softmax
//   across chunks, O rescaled by alpha = exp(m_old - m_new) once per chunk.
//
// int4 handling (FUSED=1, default): K and V are never unpacked to DTCM. The
// MAC loop loads the packed B bytes and splits them with shifts straight into
// the Zvt operand registers (mac4_p4 / mac1_p4). Q is unpacked once per tile
// (unpack_i4_rows, 512 B); P is int8 by construction. FUSED=0 unpacks each K/V
// group to an int8 copy first (unpack_i4_rows), for the unpack-cost number.
// INT8=1 is the same kernel with int8 storage (no unpack), for comparison.
//
// Layouts (written that way by the producing qkv GEMM; bytes per key fixed):
//   Q  [head][qtile][DK/2][16]  byte (i,q) = Q[i][q] | Q[i+DK/2][q] << 4
//   K  [head][64-key group][DK/4 steps][L0 64 keys][L1 64 keys], tail groups
//      as 16-key blocks [DK/4][L0 16][L1 16]; per 4-dim step L0 = d0 | d2<<4,
//      L1 = d1 | d3<<4 (FUSED=0: [DK/2][g] with (d, d+DK/2) pairs)
//   V  [head][4-key step][L0 64 dims][L1 64 dims], L0 = k0 | k2<<4,
//      L1 = k1 | k3<<4 (FUSED=0: per group [g/2][64], (k, k+g/2) pairs)
//   O  [head][qtile][DV/2][16]  dims (d, d+DV/2) packed
//
//   -DN=400|1600 -DDK=32|64 -DHEADS=2 -DRESIDENT=0|1 -DFUSED=0|1 -DINT8=0|1
//   -DKC=<keys per softmax chunk> -DEXP2=0|1
#include "ops.hpp"

using namespace ops;

#ifndef N
#define N 400
#endif
#ifndef DK
#define DK 32 // Ultralytics PSA: key_dim = head_dim * attn_ratio(0.5) = 32
#endif
#ifndef DV
#define DV 64 // head_dim = C / num_heads = 128 / 2
#endif
#ifndef HEADS
#define HEADS 2
#endif
#ifndef KC
#define KC (N <= 800 ? N : 800)
#endif
#ifndef RESIDENT
#define RESIDENT 0
#endif
#ifndef INT8
#define INT8 0
#endif
#ifndef FUSED
#define FUSED 1 // unpack int4 inside the MAC loop (no int8 copy in DTCM)
#endif
#if INT8
#undef FUSED
#define FUSED 0
#endif
#ifndef EXP2
#define EXP2 0 // 1: 16-entry in-register LUT (vrgather) + shift (slower here)
#endif
#ifndef NATIVE4
#define NATIVE4 0 // proxy for a native int4 Zvt type (k=8 per vtmm, 512 MAC/cyc)
#endif
#ifndef PINT4
#define PINT4 1 // with NATIVE4: P quantized to u4 (0..15) so PV is int4 x int4 too
#endif
#if NATIVE4 && INT8
#error "NATIVE4 is an int4 mode"
#endif
#ifndef SSHIFT
#define SSHIFT 4 // int32 score -> int8 logit (softmax scale folded in)
#endif

constexpr uint32_t kN = N, kDK = DK, kDV = DV, kHeads = HEADS, kKC = KC;
constexpr uint32_t kQTiles = kN / 16;
constexpr uint32_t kChunks = kN / kKC;
constexpr uint32_t kKG = 64; // keys per K group (mac4: 4 tiles x 16 keys)
constexpr uint32_t kVG = 32; // keys per V group
static_assert(kN % 16 == 0 && kKC % 16 == 0 && kN % kKC == 0, "shapes");
static_assert(kDV == 64, "mac4 B rows are 64 bytes: one V row = 64 dims");
static_assert(kDK % 8 == 0, "DK");
constexpr bool kMulti = kChunks > 1;

// Bytes per key / per token as stored.
constexpr uint32_t kKB = INT8 ? kDK : kDK / 2;
constexpr uint32_t kVB = INT8 ? kDV : kDV / 2;
constexpr uint32_t kQTileBytes = 16 * kKB;
constexpr uint32_t kOTileBytes = 16 * kVB;

// PSRAM map.
constexpr uint32_t kQAddr = kRam;
constexpr uint32_t kKAddr = kQAddr + kHeads * kN * kKB;
constexpr uint32_t kVAddr = kKAddr + kHeads * kN * kKB;
constexpr uint32_t kOAddr = kVAddr + kHeads * kN * kVB;

constexpr uint32_t cmax(uint32_t a, uint32_t b) { return a > b ? a : b; }
constexpr uint32_t cmin(uint32_t a, uint32_t b) { return a < b ? a : b; }

// ---- DTCM buffers ------------------------------------------------------------
alignas(64) static int8_t sp[kKC * 16]; // S (int8 logits) then P, [key][query]
#if INT8
static int8_t* q8; // Q^T tile, [dim][query]: the current qbuf
#elif !NATIVE4
alignas(64) static int8_t q8[kDK * 16]; // Q^T tile, [dim][query]
#endif
alignas(64) static uint8_t qbuf[2][kQTileBytes]; // Q prefetch (double buffer)
alignas(64) static uint8_t obuf[2][kOTileBytes];
alignas(64) static uint8_t exp_table[256];
alignas(64) static int8_t mrep[64];
alignas(64) static uint8_t exp2_tab[64]; // 127 * 2^(-i/16), i = 0..15, x4
#if !INT8
#endif
#if !INT8 && !FUSED
alignas(64) static int8_t opnd[cmax(kKG * kDK, kVG * kDV)]; // unpacked group
#endif
#if RESIDENT
alignas(64) static uint8_t kres[kN * kKB];
alignas(64) static uint8_t vres[kN * kVB];
#else
alignas(64) static uint8_t sbuf[2][cmax(kKG * kKB, kVG * kVB)];
#endif
#if KC < N
alignas(64) static int32_t out_acc[kDV * 16];
#endif


// ---- Fused int4 MAC ------------------------------------------------------------
// B operand packed for the k-step: per step, L0 = r0 | r2 << 4 and
// L1 = r1 | r3 << 4 (r0..r3 = the step's 4 B rows). Two 16 B loads + six
// shifts make the 4 int8 B rows in v16..v22, instead of 4 loads from an int8
// copy. A (Q or P, int8) as in mac4.
// mac4_p4: per step 128 B = L0[64] L1[64] (4 groups x 16 lanes), 4 vtmm.
#define ATT_A_LOADS                                                            \
    "vle8.v v8, (%0)\n\tadd %0, %0, %4\n\t"                                    \
    "vle8.v v10, (%0)\n\tadd %0, %0, %4\n\t"                                   \
    "vle8.v v12, (%0)\n\tadd %0, %0, %4\n\t"                                   \
    "vle8.v v14, (%0)\n\tadd %0, %0, %4\n\t"
#define ATT_UNPACK_B                                                           \
    "vsll.vi v16, v24, 4\n\tvsra.vi v16, v16, 4\n\tvsra.vi v20, v24, 4\n\t"    \
    "vsll.vi v18, v26, 4\n\tvsra.vi v18, v18, 4\n\tvsra.vi v22, v26, 4\n\t"
#define ATT_LB(g, rd)                                                          \
    "addi %3, %1, " #g "*16\n\tvle8.v v24, (%3)\n\taddi %3, %3, 64\n\t"        \
    "vle8.v v26, (%3)\n\t" ATT_UNPACK_B ".insn r 0x77, 0, 0x79, " #rd           \
    ", x16, x8\n\t"
E2E_INLINE void mac4_p4(const void* a_rows, uint32_t a_stride, const void* b, uint32_t steps) {
    const void* bp;
    asm volatile("1:\n\t" ATT_A_LOADS ATT_LB(0, x1) ATT_LB(1, x9) ATT_LB(2, x17) ATT_LB(3, x25)
                 "addi %1, %1, 128\n\taddi %2, %2, -1\n\tbnez %2, 1b"
                 : "+r"(a_rows), "+r"(b), "+r"(steps), "=&r"(bp)
                 : "r"(a_stride)
                 : "v8", "v10", "v12", "v14", "v16", "v18", "v20", "v22", "v24", "v26", "memory");
}
// mac1_p4: one 16-lane group, per step 32 B = L0[16] L1[16], vtmm into mt0.
E2E_INLINE void mac1_p4(const void* a_rows, uint32_t a_stride, const void* b, uint32_t steps) {
    const void* bp;
    asm volatile("1:\n\t" ATT_A_LOADS
                 "vle8.v v24, (%1)\n\taddi %3, %1, 16\n\tvle8.v v26, (%3)\n\t" ATT_UNPACK_B
                 ".insn r 0x77, 0, 0x79, x1, x16, x8\n\t"
                 "addi %1, %1, 32\n\taddi %2, %2, -1\n\tbnez %2, 1b"
                 : "+r"(a_rows), "+r"(b), "+r"(steps), "=&r"(bp)
                 : "r"(a_stride)
                 : "v8", "v10", "v12", "v14", "v16", "v18", "v20", "v22", "v24", "v26", "memory");
}

// Walk keys [k0, k1) in groups of at most gmax, handing body(data, g, k) each
// group's bytes (bpk bytes per key): DMA double-buffered from PSRAM at src, or
// straight out of the resident copy.
template <typename F>
static inline void walk(const uint8_t* resident, uint32_t src, uint32_t bpk,
                        uint32_t k0, uint32_t k1, uint32_t gmax, F body) {
#if RESIDENT
    (void)src;
    for (uint32_t k = k0; k < k1;) {
        const uint32_t g = cmin(gmax, k1 - k);
        body(resident + k * bpk, g, k);
        k += g;
    }
#else
    (void)resident;
    uint32_t buf = 0, k = k0, g = cmin(gmax, k1 - k0);
    dma(src + k * bpk, addr(sbuf[0]), g * bpk);
    while (k < k1) {
        dma_wait();
        const uint32_t kn = k + g;
        const uint32_t gn = kn < k1 ? cmin(gmax, k1 - kn) : 0;
        if (kn < k1) dma(src + kn * bpk, addr(sbuf[buf ^ 1]), gn * bpk);
        body(sbuf[buf], g, k);
        k = kn;
        g = gn;
        buf ^= 1;
    }
#endif
}

// The 16 columns of int32 tile `acc` (16 keys x 16 queries) -> int8 logits at
// dst (16 B per key). One asm loop: vtmv into v4, then narrow straight from
// v4 (no vmv4r copy).
#define ATT_STR(x) #x
#define ATT_XSTR(x) ATT_STR(x)
E2E_INLINE void readout16(uint32_t acc, int8_t* dst) {
    uint32_t sel = acc_tile(acc) | kColumnSlice, n = 16;
    asm volatile("csrwi vxrm, 2\n\t"
                 "1:\n\t"
                 ".insn r 0x57, 7, 0x42, x0, %3, %4\n\t"
                 ".insn r 0x57, 7, 0x44, x0, %5, x0\n\t"
                 ".insn r 0x57, 6, 0x21, x4, %0, x31\n\t"
                 "vsetivli zero, 16, e16, m2, ta, ma\n\t"
                 "vnclip.wi v24, v4, " ATT_XSTR(SSHIFT) "\n\t"
                 "vsetivli zero, 16, e8, m1, ta, ma\n\t"
                 "vnclip.wi v24, v24, 0\n\t"
                 "vse8.v v24, (%1)\n\t"
                 "addi %1, %1, 16\n\taddi %0, %0, 1\n\taddi %2, %2, -1\n\tbnez %2, 1b"
                 : "+r"(sel), "+r"(dst), "+r"(n)
                 : "r"((16U << 10) | (1U << 5) | 1U), "r"(2U << 3), "r"(16U)
                 : "v4", "v5", "v6", "v7", "v24", "v25", "vl", "vtype", "memory");
}

// Softmax over the chunk's int8 logits in sp: P = exp(s - m) (0..127) in place.
// Returns the per-query row sum of this chunk; m_io is the running max (in:
// previous max, out: new max); alpha_out = exp(m_old - m_new).
static inline vuint32m4_t softmax_chunk(vint8m1_t& m_io, vuint32m4_t& alpha_out) {
    vint8m4_t mx = __riscv_vle8_v_i8m4(sp, 64);
    for (uint32_t i = 1; i < kKC / 4; ++i) {
        mx = __riscv_vmax_vv_i8m4(mx, __riscv_vle8_v_i8m4(sp + 64 * i, 64), 64);
    }
    vint8m1_t m = __riscv_vmax_vv_i8m1(
        __riscv_vmax_vv_i8m1(__riscv_vget_v_i8m4_i8m1(mx, 0), __riscv_vget_v_i8m4_i8m1(mx, 1), 16),
        __riscv_vmax_vv_i8m1(__riscv_vget_v_i8m4_i8m1(mx, 2), __riscv_vget_v_i8m4_i8m1(mx, 3), 16),
        16);
    m = __riscv_vmax_vv_i8m1(m, m_io, 16);
    const vuint8m1_t agap = __riscv_vreinterpret_v_i8m1_u8m1(__riscv_vsub_vv_i8m1(m, m_io, 16));
    alpha_out = __riscv_vzext_vf4_u32m4(__riscv_vluxei8_v_u8m1(exp_table, agap, 16), 16);
    m_io = m;
    for (uint32_t i = 0; i < 4; ++i) __riscv_vse8_v_i8m1(mrep + 16 * i, m, 16);
    const vint8m4_t m4 = __riscv_vle8_v_i8m4(mrep, 64);
    vuint16m8_t sum = __riscv_vmv_v_x_u16m8(0, 64);
#if EXP2
    const vuint8m4_t tab = __riscv_vle8_v_u8m4(exp2_tab, 64);
#endif
    for (uint32_t i = 0; i < kKC / 4; ++i) {
        int8_t* row = sp + 64 * i;
        const vuint8m4_t gap = __riscv_vreinterpret_v_i8m4_u8m4(
            __riscv_vsub_vv_i8m4(m4, __riscv_vle8_v_i8m4(row, 64), 64));
#if EXP2
        // 127 * 2^(-gap/16) = tab[gap & 15] >> (gap >> 4)  (logit scale has
        // log2(e) folded in, so this is the same softmax as the exp LUT).
        const vuint8m4_t p = __riscv_vsrl_vv_u8m4(
            __riscv_vrgather_vv_u8m4(tab, __riscv_vand_vx_u8m4(gap, 15, 64), 64),
            __riscv_vminu_vx_u8m4(__riscv_vsrl_vx_u8m4(gap, 4, 64), 7, 64), 64);
#else
        const vuint8m4_t p = __riscv_vluxei8_v_u8m4(exp_table, gap, 64);
#endif
#if NATIVE4 && PINT4
        // P -> u4 (0..15); keys (4i, 4i+1) and (4i+2, 4i+3) packed per byte,
        // written in place at row pair 2i, 2i+1 ([keys/2][16]).
        const vuint8m4_t p4 = __riscv_vsrl_vx_u8m4(p, 3, 64);
        sum = __riscv_vwaddu_wv_u16m8(sum, p4, 64);
        uint8_t* prow = reinterpret_cast<uint8_t*>(sp) + 32 * i;
        __riscv_vse8_v_u8m1(prow, __riscv_vor_vv_u8m1(__riscv_vget_v_u8m4_u8m1(p4, 0),
            __riscv_vsll_vx_u8m1(__riscv_vget_v_u8m4_u8m1(p4, 1), 4, 16), 16), 16);
        __riscv_vse8_v_u8m1(prow + 16, __riscv_vor_vv_u8m1(__riscv_vget_v_u8m4_u8m1(p4, 2),
            __riscv_vsll_vx_u8m1(__riscv_vget_v_u8m4_u8m1(p4, 3), 4, 16), 16), 16);
#else
        sum = __riscv_vwaddu_wv_u16m8(sum, p, 64);
        __riscv_vse8_v_u8m4(reinterpret_cast<uint8_t*>(row), p, 64);
#endif
    }
    return __riscv_vadd_vv_u32m4(
        __riscv_vwaddu_vv_u32m4(__riscv_vget_v_u16m8_u16m2(sum, 0), __riscv_vget_v_u16m8_u16m2(sum, 1), 16),
        __riscv_vwaddu_vv_u32m4(__riscv_vget_v_u16m8_u16m2(sum, 2), __riscv_vget_v_u16m8_u16m2(sum, 3), 16),
        16);
}

// O[dim] (16 query lanes, int32) / rowsum -> int4 (or int8), into obuf[ob].
template <typename Get>
static inline void epilogue(Get get, vuint32m4_t rsum, uint32_t ob) {
    const vint32m4_t recip = __riscv_vreinterpret_v_u32m4_i32m4(
        __riscv_vdivu_vv_u32m4(__riscv_vmv_v_x_u32m4(1U << 16, 16), rsum, 16));
    auto norm = [&](uint32_t d) {
        return __riscv_vsra_vx_i32m4(__riscv_vmul_vv_i32m4(get(d), recip, 16), 16, 16);
    };
#if INT8
    for (uint32_t d = 0; d < kDV; ++d) {
        __riscv_vse8_v_i8m1(reinterpret_cast<int8_t*>(obuf[ob]) + d * 16, to_i8(norm(d)), 16);
    }
#else
    for (uint32_t d = 0; d < kDV / 2; ++d) {
        const vint32m4_t lo = norm(d);
        __riscv_vse8_v_u8m1(obuf[ob] + d * 16, pack_i4(lo, norm(d + kDV / 2)), 16);
    }
#endif
}

E2E_INLINE vint32m4_t acc_col(uint32_t d) {
    return vtmv(acc_tile(d / 16) | kColumnSlice | (d % 16));
}

int main() {
    // exp(-i/16) * 127: the softmax LUT (built before the ROI; it would be a
    // 256 B flash load per layer).
    uint32_t decay = 127U << 16;
    for (uint32_t i = 0; i < 256; ++i) {
        exp_table[i] = static_cast<uint8_t>(decay >> 16);
        decay = (decay >> 16) * 61565U;
    }
    uint32_t e2 = 127U << 8; // 127 * 2^(-i/16), Q8
    for (uint32_t i = 0; i < 16; ++i) {
        for (uint32_t r = 0; r < 4; ++r) exp2_tab[16 * r + i] = static_cast<uint8_t>(e2 >> 8);
        e2 = (e2 * 62757U) >> 16; // 65536 * 2^(-1/16)
    }
    matrix_on();
    roi_begin();
    dma(kQAddr, addr(qbuf[0]), kQTileBytes);
    for (uint32_t h = 0; h < kHeads; ++h) {
        const uint32_t kbase = kKAddr + h * kN * kKB;
        const uint32_t vbase = kVAddr + h * kN * kVB;
#if RESIDENT
        dma(kbase, addr(kres), kN * kKB);
        dma(vbase, addr(vres), kN * kVB);
        dma_wait();
        mark(); // resident K/V load
        const uint8_t* kr = kres;
        const uint8_t* vr = vres;
#else
        const uint8_t* kr = nullptr;
        const uint8_t* vr = nullptr;
#endif
        for (uint32_t t = 0; t < kQTiles; ++t) {
            const uint32_t ob = (h * kQTiles + t) & 1;
            // Q tile: [DK][16] int8.
            // Q tile (prefetched one tile ahead): [DK][16] int8.
            const uint32_t qt = h * kQTiles + t;
            dma_wait();
#if INT8
            q8 = reinterpret_cast<int8_t*>(qbuf[qt & 1]);
#elif NATIVE4
            // Native int4: Q^T packed [DK/2][16] is the A operand as is.
            const auto* qa = reinterpret_cast<const int8_t*>(qbuf[qt & 1]);
#else
            unpack_i4_rows(qbuf[qt & 1], q8, kQTileBytes);
#endif
            if (qt > 0) // previous tile's O: issued now so this wait never blocks on it
                dma(addr(obuf[(qt - 1) & 1]), kOAddr + (qt - 1) * kOTileBytes, kOTileBytes);
            if (qt + 1 < kHeads * kQTiles)
                dma(kQAddr + (qt + 1) * kQTileBytes, addr(qbuf[(qt + 1) & 1]), kQTileBytes);
            mark(); // Q load
            vint8m1_t m_run = __riscv_vmv_v_x_i8m1(-128, 16);
            vuint32m4_t rsum = __riscv_vmv_v_x_u32m4(0, 16);
            for (uint32_t c = 0; c < kChunks; ++c) {
                const uint32_t c0 = c * kKC;
                // 1. S = Q K^T over the chunk, int8 logits into sp.
                walk(kr, kbase, kKB, c0, c0 + kKC, kKG, [&](const uint8_t* data, uint32_t g, uint32_t k) {
#if INT8 || FUSED
                    const auto* k8 = reinterpret_cast<const int8_t*>(data);
#else
                    unpack_i4_rows(data, opnd, g * kDK / 2); // -> [DK][g]
                    const int8_t* k8 = opnd;
#endif
                    mark(); // K wait + unpack
                    const uint32_t kl = k - c0;
                    if (g == kKG) {
                        vtzero(true);
                        config_int8(16);
#if NATIVE4
                        // PROXY: packed bytes fed as int8, DK/8 steps = k=8 per vtmm.
                        mac4(qa, 16, k8, kDK / 8);
#elif FUSED
                        mac4_p4(q8, 16, k8, kDK / 4);
#else
                        mac4(q8, 16, k8, kDK / 4);
#endif
                        mark(); // QK mac
                        for (uint32_t tt = 0; tt < 4; ++tt) readout16(tt, sp + (kl + 16 * tt) * 16);
                    } else {
                        for (uint32_t b = 0; b < g / 16; ++b) {
                            vtzero();
                            config_int8(16);
#if NATIVE4
                            mac<true>(qa, 16, k8 + 16 * b, g, kDK / 8); // [DK/2][g]
#elif FUSED
                            mac1_p4(q8, 16, k8 + b * 16 * kKB, kDK / 4); // 16-key blocks
#else
                            mac<true>(q8, 16, k8 + 16 * b, g, kDK / 4);
#endif
                            readout16(0, sp + (kl + 16 * b) * 16);
                        }
                        mark(); // QK mac (tail; readout included)
                    }
                    mark(); // S readout
                });
                // 2. Softmax over the whole chunk.
                vuint32m4_t alpha;
                const vuint32m4_t csum = softmax_chunk(m_run, alpha);
                mark(); // softmax
                // 3. O += P V over the chunk, in the accumulators.
                vtzero(true);
                walk(vr, vbase, kVB, c0, c0 + kKC, kVG, [&](const uint8_t* data, uint32_t g, uint32_t k) {
#if INT8 || FUSED
                    const auto* v8 = reinterpret_cast<const int8_t*>(data);
#else
                    unpack_i4_rows(data, opnd, g * kDV / 2); // -> [g][64]
                    const int8_t* v8 = opnd;
#endif
                    mark(); // V wait + unpack
                    config_int8(16);
#if NATIVE4 && PINT4
                    // PROXY: P packed u4 [keys/2][16], V packed [g/2][64]: g/8 steps.
                    mac4(sp + (k - c0) * 8, 16, v8, g / 8);
#elif FUSED
                    mac4_p4(sp + (k - c0) * 16, 16, v8, g / 4);
#else
                    mac4(sp + (k - c0) * 16, 16, v8, g / 4);
#endif
                    mark(); // PV mac
                });
#if KC < N
                // Online softmax across chunks: acc = acc*alpha + O_chunk.
                const vint32m4_t al = __riscv_vreinterpret_v_u32m4_i32m4(alpha);
                for (uint32_t d = 0; d < kDV; ++d) {
                    int32_t* row = out_acc + d * 16;
                    vint32m4_t v = acc_col(d);
                    if (c) {
                        v = __riscv_vadd_vv_i32m4(
                            v,
                            __riscv_vsra_vx_i32m4(__riscv_vmul_vv_i32m4(__riscv_vle32_v_i32m4(row, 16), al, 16), 7, 16),
                            16);
                    }
                    __riscv_vse32_v_i32m4(row, v, 16);
                }
                rsum = __riscv_vadd_vv_u32m4(__riscv_vsrl_vx_u32m4(__riscv_vmul_vv_u32m4(rsum, alpha, 16), 7, 16), csum, 16);
                mark(); // chunk rescale
#else
                (void)alpha;
                rsum = csum;
#endif
            }
#if KC < N
            epilogue([&](uint32_t d) { return __riscv_vle32_v_i32m4(out_acc + d * 16, 16); }, rsum, ob);
#else
            epilogue([&](uint32_t d) { return acc_col(d); }, rsum, ob);
#endif
            mark(); // epilogue (O write issued after the next tile's Q wait)
        }
    }
    dma(addr(obuf[(kHeads * kQTiles - 1) & 1]), kOAddr + (kHeads * kQTiles - 1) * kOTileBytes, kOTileBytes);
    dma_wait();
    roi_end();
    return 0;
}
