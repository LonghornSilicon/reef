// Depthwise 3x3 convolution on the Zvt MATRIX engine (stride 1, pad 1, W4A4):
// the "block-Toeplitz" mapping, to measure how badly a K=9 operator fits.
//
// Depthwise has no channel reduction, so put something else on the matrix
// dims. Per channel c, one 16x16 output tile = 16 output rows (m) x 16 output
// pixels (n):
//   C[m][n] = sum_k A[k][m] * B[k][n],  k = (dx, i), i = 0..19, dx = 0..2
//   A[(dx,i)][m] = w[c][i-m][dx] if 0 <= i-m <= 2 else 0  (banded Toeplitz,
//                  depends only on the channel: 60 x 16 int8, built offline)
//   B[(dx,i)][n] = in[c][y0-1+i][x0-1+dx+n]               (shifted input rows)
// K = 3 x 20 = 60 (18 rows needed, padded to 20 so each dx block is whole
// k-steps), so 15 vtmm per tile for 16*16*9 = 2304 useful MACs out of
// 15*4*256 = 15360 issued: 15% of the array at best.
// mac4 shares A (the weights) across 4 x-tiles: per k-step 4 A + 16 B loads.
//
// The obvious alternative, block-diagonal weights (K = 16 channels x 9 taps,
// B[(c,t)][n] = w[c][t] if n == c), uses 1/16 of every vtmm (6%) and cannot
// share A across accumulators: analytic only, see the report.
//
// Layout: channel-planar int4, row-pair packed (rows 2p, 2p+1 in one byte),
// the same as dwconv_planar.cc. Per channel: DMA the packed plane (double-
// buffered), unpack once to an int8 plane with zero margins, stream the
// Toeplitz A (int4, unpacked once per channel), run 5 bands x (4 + 1)
// x-tiles, epilogue via vtmv row slices, requant, pack row pairs.
//
//   -DH= -DW= -DC= (H % 16 == 0, W = 64 + 16), -DCH=n channels to compute
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
#ifndef CH
#define CH C
#endif
static_assert(H % 16 == 0 && W == 80, "this measurement covers 80x80 only");

constexpr uint32_t kPacked = H / 2 * W;        // packed plane bytes
constexpr uint32_t kPitch = W + 32;            // int8 plane row: 16 B margins
constexpr uint32_t kPlaneRows = H + 5;         // 1 zero row above, 4 below
constexpr uint32_t kKRows = 20;                // k rows per dx block
constexpr uint32_t kTz = 3 * kKRows * 16;      // Toeplitz A bytes (int8)

constexpr uint32_t kFlashTz = kFlash;          // [C][kTz/2] int4
constexpr uint32_t kFlashBias = kFlash + C * kTz / 2;
constexpr uint32_t kRamIn = kRam;
constexpr uint32_t kRamOut = kRam + C * kPacked;

static uint8_t packed[2][kPacked];
static uint8_t tz_packed[2][kTz / 2];
static int8_t tz[kTz];
static int8_t plane[kPlaneRows * kPitch];
static uint8_t outp[2][kPacked];
static int32_t qparams[2 * C]; // bias, multiplier per channel

// mac4 with a B row stride (npu.hpp's mac4 fixes it at 64).
E2E_INLINE void mac4s(const void* a_rows, uint32_t a_stride, const void* b_rows,
                      uint32_t b_stride, uint32_t steps) {
    const void* b_ptr;
#define LOAD_B(g, rd)                                                          \
    "addi %3, %1, " #g "*16\n\tvle8.v v16, (%3)\n\tadd %3, %3, %5\n\t"         \
    "vle8.v v18, (%3)\n\tadd %3, %3, %5\n\tvle8.v v20, (%3)\n\t"               \
    "add %3, %3, %5\n\tvle8.v v22, (%3)\n\t"                                   \
    ".insn r 0x77, 0, 0x79, " #rd ", x16, x8\n\t"
    asm volatile("1:\n\t"
                 "vle8.v v8, (%0)\n\tadd %0, %0, %4\n\t"
                 "vle8.v v10, (%0)\n\tadd %0, %0, %4\n\t"
                 "vle8.v v12, (%0)\n\tadd %0, %0, %4\n\t"
                 "vle8.v v14, (%0)\n\tadd %0, %0, %4\n\t"
                 LOAD_B(0, x1) LOAD_B(1, x9) LOAD_B(2, x17) LOAD_B(3, x25)
                 "add %1, %1, %6\n\t"
                 "addi %2, %2, -1\n\t"
                 "bnez %2, 1b"
                 : "+r"(a_rows), "+r"(b_rows), "+r"(steps), "=&r"(b_ptr)
                 : "r"(a_stride), "r"(b_stride), "r"(4 * b_stride)
                 : "v8", "v10", "v12", "v14", "v16", "v18", "v20", "v22",
                   "memory");
#undef LOAD_B
}

// Requantize the 16x16 tile in accumulator acc (rows = output rows y0..,
// cols = pixels x0..) and store it as 8 packed row pairs.
static void epilogue(uint32_t acc, uint8_t* out, uint32_t y0, uint32_t x0,
                     int32_t b, int32_t m) {
    for (uint32_t j = 0; j < 16; j += 2) {
        vint32m4_t r0 = vtmv(acc_tile(acc) | j);
        vint32m4_t r1 = vtmv(acc_tile(acc) | (j + 1));
        r0 = __riscv_vsra_vx_i32m4(__riscv_vmul_vx_i32m4(__riscv_vadd_vx_i32m4(r0, b, 16), m, 16), 16, 16);
        r1 = __riscv_vsra_vx_i32m4(__riscv_vmul_vx_i32m4(__riscv_vadd_vx_i32m4(r1, b, 16), m, 16), 16, 16);
        __riscv_vse8_v_u8m1(out + (y0 + j) / 2 * W + x0, pack_i4(r0, r1), 16);
    }
}

static void unpack_plane(const uint8_t* src) {
    for (uint32_t p = 0; p < H / 2; ++p) {
        const vint8m8_t b = __riscv_vreinterpret_v_u8m8_i8m8(__riscv_vle8_v_u8m8(src + p * W, W));
        __riscv_vse8_v_i8m8(plane + (2 * p + 1) * kPitch + 16,
                            __riscv_vsra_vx_i8m8(__riscv_vsll_vx_i8m8(b, 4, W), 4, W), W);
        __riscv_vse8_v_i8m8(plane + (2 * p + 2) * kPitch + 16, __riscv_vsra_vx_i8m8(b, 4, W), W);
    }
}

static void issue(uint32_t c) {
    dma(kRamIn + c * kPacked, addr(packed[c & 1]), kPacked);
    dma(kFlashTz + c * kTz / 2, addr(tz_packed[c & 1]), kTz / 2);
}

int main() {
    matrix_on();
    roi_begin();
    dma(kFlashBias, addr(qparams), sizeof(qparams));
    issue(0);
    for (uint32_t c = 0; c < CH; ++c) {
        dma_wait(); // plane c in, output plane c-1's buffer drained
        if (c > 0) dma(addr(outp[(c - 1) & 1]), kRamOut + (c - 1) * kPacked, kPacked);
        if (c + 1 < CH) issue(c + 1);
        unpack_plane(packed[c & 1]);
        unpack_i4_rows(tz_packed[c & 1], tz, kTz / 2);
        const int32_t b = qparams[2 * c], m = qparams[2 * c + 1];
        uint8_t* out = outp[c & 1];
        for (uint32_t y0 = 0; y0 < H; y0 += 16) {
            // Input row y0-1+i lives at plane row y0+i; pixel x at column 16+x.
            const int8_t* b0 = plane + y0 * kPitch + 16 - 1;
            vtzero(true);
            config_int8(16);
            for (uint32_t dx = 0; dx < 3; ++dx)
                mac4s(tz + dx * kKRows * 16, 16, b0 + dx, kPitch, kKRows / 4);
            for (uint32_t g = 0; g < 4; ++g) epilogue(g, out, y0, 16 * g, b, m);
            // Fifth x-tile (pixels 64..79) on one accumulator.
            vtzero();
            config_int8(16);
            for (uint32_t dx = 0; dx < 3; ++dx)
                mac<true>(tz + dx * kKRows * 16, 16, b0 + 64 + dx, kPitch, kKRows / 4);
            epilogue(0, out, y0, 64, b, m);
        }
    }
    dma_wait();
    dma(addr(outp[(CH - 1) & 1]), kRamOut + (CH - 1) * kPacked, kPacked);
    dma_wait();
    roi_end();
    return 0;
}
