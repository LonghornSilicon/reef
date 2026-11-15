// DCGAN generator layer 1: ConvTranspose2d(100 -> 512, k4, s1, p0) on a 1x1
// input = a GEMV z[100] x W[100 x 8192] (column n = och*16 + ky*4 + kx), W4A4.
//
// Batch 1, so there is one GEMM row: tm = 1. A = z (stride 1: lane 0 of each
// A row is z[k]); B = weights, 64 columns (4 och x 16 pixels) per group, the
// linear.cc mac4 layout. Each weight is used exactly once (no reuse at batch
// 1), so the layer is a weight stream: flash -> DTCM (packed, double-buffered,
// 3.2 KB per group + 32 B of folded-BN params), unpack once, mac4.
// Epilogue: accumulator g = output channel 4*grp+g, row 0 = its 4x4 map;
// *mult + bias >> shift, ReLU, pack channel pairs to int4, scatter the 16
// pixels into layer 2's padded plane (pitch 5, see convt.cc) and DMA it out.
//   -DNZ=100 -DNOUT=512
#include "ops.hpp"

using namespace ops;

#ifndef NZ
#define NZ 100
#endif
#ifndef NOUT
#define NOUT 512
#endif
#ifndef FUSED
#define FUSED 1 // unpack int4 B rows straight into v16..v22 in the MAC loop
#endif

constexpr uint32_t kGroups = NOUT / 4;               // 64 columns = 4 och
constexpr uint32_t kPacked = NZ * 64 / 2;            // 3200 B per group
constexpr uint32_t kGroupBytes = kPacked + 4 * 8;    // + mult, bias per och
constexpr uint32_t kPlane = 32 + 2 * 5 + 2;          // layer-2 plane (convt.cc kLin)
constexpr uint32_t kShift = 16;
constexpr uint32_t kOutBase = kRam;                  // layer 2 input
static_assert(NZ % 4 == 0, "K steps");

static uint8_t wpk[2][kGroupBytes] __attribute__((aligned(16)));
#if !FUSED
static int8_t w8[NZ * 64] __attribute__((aligned(16)));
#else
// Packed group layout: per k-quad, per accumulator g: 32 B = rows k0/k1
// (low/high nibble) then k2/k3. 2 loads + 6 shifts per accumulator.
#define FC_B(rd)                                                               \
    "vle8.v v24, (%1)\n\taddi %1, %1, 16\n\tvle8.v v26, (%1)\n\taddi %1, %1, 16\n\t" \
    "vsll.vi v16, v24, 4\n\tvsra.vi v16, v16, 4\n\tvsra.vi v18, v24, 4\n\t"      \
    "vsll.vi v20, v26, 4\n\tvsra.vi v20, v20, 4\n\tvsra.vi v22, v26, 4\n\t"      \
    ".insn r 0x77, 0, 0x79, " #rd ", x16, x8\n\t"
static void mac4_fused(const int8_t* a, const uint8_t* b, uint32_t steps) {
    asm volatile("1:\n\t"
                 "vle8.v v8, (%0)\n\taddi %0, %0, 1\n\t"
                 "vle8.v v10, (%0)\n\taddi %0, %0, 1\n\t"
                 "vle8.v v12, (%0)\n\taddi %0, %0, 1\n\t"
                 "vle8.v v14, (%0)\n\taddi %0, %0, 1\n\t" FC_B(x1) FC_B(x9)
                     FC_B(x17) FC_B(x25) "addi %2, %2, -1\n\tbnez %2, 1b"
                 : "+r"(a), "+r"(b), "+r"(steps)
                 :
                 : "v8", "v10", "v12", "v14", "v16", "v18", "v20", "v22", "v24",
                   "v26", "memory");
}
#endif
static uint8_t zpk[64] __attribute__((aligned(16)));
static int8_t z8[128 + 16] __attribute__((aligned(16)));
static uint8_t stage[2][48] __attribute__((aligned(16)));
// Pixel i of a 4x4 map -> its byte in the padded plane.
static const uint8_t kScatter[16] = {6,  7,  8,  9,  11, 12, 13, 14,
                                     16, 17, 18, 19, 21, 22, 23, 24};

static void issue(uint32_t grp) {
    dma(kFlash + grp * kGroupBytes, addr(wpk[grp & 1]), kGroupBytes);
}

static inline vint32m4_t requant(vint32m4_t v, const int32_t* p, uint32_t g) {
    v = __riscv_vmul_vx_i32m4(v, p[g], 16);
    v = __riscv_vsra_vx_i32m4(__riscv_vadd_vx_i32m4(v, p[4 + g], 16), kShift, 16);
    return __riscv_vmax_vx_i32m4(v, 0, 16);
}

int main() {
    matrix_on();
    roi_begin();
    issue(0);
    dma(kRam + (6u << 20), addr(zpk), NZ / 2); // latent z, packed
    dma_wait();
    unpack_i4_rows(zpk, z8, 64);               // z8[0..100) (row pair split)
    const vuint8m1_t scatter = __riscv_vle8_v_u8m1(kScatter, 16);
    for (uint32_t grp = 0; grp < kGroups; ++grp) {
        dma_wait();
        if (grp + 1 < kGroups) issue(grp + 1);
        vtzero(true);
#if FUSED
        config_int8(1);
        mac4_fused(z8, wpk[grp & 1], NZ / 4);
#else
        unpack_i4_rows(wpk[grp & 1], w8, kPacked);
        config_int8(1);
        mac4(z8, 1, w8, NZ / 4);
#endif
        const auto* p = reinterpret_cast<const int32_t*>(wpk[grp & 1] + kPacked);
        for (uint32_t cp = 0; cp < 2; ++cp) {
            const vint32m4_t y0 = requant(vtmv(acc_tile(2 * cp)), p, 2 * cp);
            const vint32m4_t y1 = requant(vtmv(acc_tile(2 * cp + 1)), p, 2 * cp + 1);
            __riscv_vsuxei8_v_u8m1(stage[cp], scatter, pack_i4(y0, y1), 16);
            dma(addr(stage[cp]), kOutBase + (grp * 2 + cp) * kPlane, kPlane);
        }
    }
    dma_wait();
    roi_end();
    return 0;
}
