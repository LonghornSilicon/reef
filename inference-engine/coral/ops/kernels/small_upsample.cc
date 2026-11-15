// Nearest-neighbour 2x upsample: H x W x C -> 2H x 2W x C, int4, PSRAM to
// PSRAM. YOLO neck: 20x20x256 and 40x40x128.
//
// Default (vector) variant, ops.hpp row-pair packing applied to pixels: a
// pixel-pair group is C bytes, byte c = channel c of pixel 2p (low nibble)
// and pixel 2p+1 (high nibble). Output pixels 2x and 2x+1 both equal input
// pixel x, so input pair g (pixels 2g, 2g+1) becomes output pairs 2g and
// 2g+1:
//   out[2g][c]   = lo(b) | lo(b) << 4     (b & 0x0F, then or with << 4)
//   out[2g+1][c] = hi(b) | hi(b) >> 4     (b & 0xF0, then or with >> 4)
// 3 vector ops per output byte, no permutes. Vertical duplication is free:
// the expanded row is DMA'd to PSRAM twice (rows 2y and 2y+1).
// Rows are double-buffered: at row y the DMA writes row y-1 (twice) and reads
// row y+1 while the vector unit expands row y.
//
//   -DH= -DW= -DC=   input shape
//   -DINT8=1         int8: horizontal dup is a C-byte pixel copy
//   -DDMA_ONLY=1     no vector work. Needs channel-pair packing (channels
//                    2k, 2k+1 in one byte), so a pixel is C/2 whole bytes:
//                    each input pixel is DMA'd from DTCM 4 times (2 columns
//                    x 2 rows) straight into the output map.
#include "ops.hpp"

using namespace ops;

#ifndef H
#define H 20
#endif
#ifndef W
#define W 20
#endif
#ifndef C
#define C 256
#endif
#ifndef INT8
#define INT8 0
#endif
#ifndef DMA_ONLY
#define DMA_ONLY 0
#endif

constexpr uint32_t kPixelBytes = INT8 ? C : C / 2; // per input pixel (avg)
constexpr uint32_t kInRow = W * kPixelBytes;
constexpr uint32_t kOutRow = 2 * kInRow;
constexpr uint32_t kIn = kRam;
constexpr uint32_t kOut = kRam + 0x100000;
static_assert(W % 2 == 0 && C % 64 == 0, "shape");

static uint8_t ibuf[2][kInRow];
#if !DMA_ONLY
static uint8_t obuf[2][kOutRow];
#endif

static void read_row(uint32_t y) { dma(kIn + y * kInRow, addr(ibuf[y & 1]), kInRow); }

#if DMA_ONLY
// Input pixel x -> output pixels 2x, 2x+1 of rows 2y, 2y+1.
static void write_row(uint32_t y) {
    const uint8_t* src = ibuf[y & 1];
    const uint32_t row0 = kOut + 2 * y * kOutRow;
    for (uint32_t x = 0; x < W; ++x) {
        const uint32_t s = addr(src + x * kPixelBytes);
        const uint32_t d = row0 + 2 * x * kPixelBytes;
        dma(s, d, kPixelBytes);
        dma(s, d + kPixelBytes, kPixelBytes);
        dma(s, d + kOutRow, kPixelBytes);
        dma(s, d + kOutRow + kPixelBytes, kPixelBytes);
    }
}
#else
static void write_row(uint32_t y) {
    dma(addr(obuf[y & 1]), kOut + 2 * y * kOutRow, kOutRow);
    dma(addr(obuf[y & 1]), kOut + (2 * y + 1) * kOutRow, kOutRow);
}

static void expand(const uint8_t* in, uint8_t* out) {
#if INT8
    for (uint32_t x = 0; x < W; ++x) { // pixel x -> pixels 2x, 2x+1
        for (uint32_t c = 0; c < C;) {
            const size_t vl = __riscv_vsetvl_e8m8(C - c);
            const vuint8m8_t v = __riscv_vle8_v_u8m8(in + x * C + c, vl);
            __riscv_vse8_v_u8m8(out + 2 * x * C + c, v, vl);
            __riscv_vse8_v_u8m8(out + (2 * x + 1) * C + c, v, vl);
            c += vl;
        }
    }
#else
    for (uint32_t g = 0; g < W / 2; ++g) { // pair g -> pairs 2g, 2g+1
        for (uint32_t c = 0; c < C;) {
            const size_t vl = __riscv_vsetvl_e8m8(C - c);
            const vuint8m8_t b = __riscv_vle8_v_u8m8(in + g * C + c, vl);
            const vuint8m8_t lo = __riscv_vand_vx_u8m8(b, 0x0F, vl);
            const vuint8m8_t hi = __riscv_vand_vx_u8m8(b, 0xF0, vl);
            __riscv_vse8_v_u8m8(out + 2 * g * C + c,
                                __riscv_vor_vv_u8m8(lo, __riscv_vsll_vx_u8m8(lo, 4, vl), vl), vl);
            __riscv_vse8_v_u8m8(out + (2 * g + 1) * C + c,
                                __riscv_vor_vv_u8m8(hi, __riscv_vsrl_vx_u8m8(hi, 4, vl), vl), vl);
            c += vl;
        }
    }
#endif
}
#endif

int main() {
    roi_begin();
    read_row(0);
    for (uint32_t y = 0; y < H; ++y) {
        dma_wait(); // row y is in; rows < y-1 are out
#if DMA_ONLY
        if (y + 1 < H) {
            read_row(y + 1);
        }
        write_row(y);
#else
        if (y > 0) {
            write_row(y - 1);
        }
        if (y + 1 < H) {
            read_row(y + 1);
        }
        expand(ibuf[y & 1], obuf[y & 1]);
#endif
    }
#if !DMA_ONLY
    dma_wait();
    write_row(H - 1);
#endif
    dma_wait();
    roi_end();
    return 0;
}
