// Global average pool: H x W x C (int4, PSRAM) -> C (int4, PSRAM).
// ResNet-18 head (7x7x512, then FC 512->1000 = a linear.cc GEMM) and a
// YOLO-size 20x20x256.
//
// Layout (ops.hpp row-pair packing, applied to pixels): a "pixel pair" group
// is C bytes; byte c holds channel c of pixel 2p (low nibble) and of pixel
// 2p+1 (high nibble). An odd pixel count pads the last pair with a zero
// pixel (its high nibbles are 0, so it adds nothing to the sum).
//
// Dataflow: the input streams PSRAM -> DTCM in 4 KB chunks, double-buffered
// (chunk i+1 in flight while chunk i is reduced). Reduction: for each block of
// 64 channels, a 64-lane int16 accumulator stays in v-registers across the
// chunk's pixel pairs; each byte is unpacked with 3 shifts and both nibbles
// are added with widening adds. int16 is exact: |sum| <= 8 * H*W <= 32768
// for H*W <= 4096. Epilogue: widen to int32, multiply by a fixed-point
// reciprocal of H*W (requant scale folded in), round, shift, clamp to int4,
// pack (channel c low nibble, channel c + C/2 high nibble: the row-pair
// convention for a 2-row x C/2 vector), DMA the C/2 bytes back to PSRAM.
//   -DH= -DW= -DC=      shape
//   -DINT8=1            int8 activations (no unpack/pack, twice the bytes)
#include "ops.hpp"

using namespace ops;

#ifndef H
#define H 7
#endif
#ifndef W
#define W 7
#endif
#ifndef C
#define C 512
#endif
#ifndef INT8
#define INT8 0
#endif

constexpr uint32_t kPixels = H * W;
constexpr uint32_t kGroupPixels = INT8 ? 1 : 2; // pixels per C-byte group
constexpr uint32_t kGroups = (kPixels + kGroupPixels - 1) / kGroupPixels;
constexpr uint32_t kInBytes = kGroups * C;
constexpr uint32_t kChunkBytes = 4096;
constexpr uint32_t kChunkGroups = kChunkBytes / C;
constexpr uint32_t kChunks = (kGroups + kChunkGroups - 1) / kChunkGroups;
constexpr uint32_t kOutBytes = INT8 ? C : C / 2;
static_assert(kChunkGroups >= 1 && C % 64 == 0, "shape");
static_assert(kPixels <= 4096, "int16 accumulator range");

// Fixed-point 1/(H*W): mean = (sum * kRecip + round) >> kShift.
constexpr uint32_t kShift = 20;
constexpr int32_t kRecip = ((1 << kShift) + kPixels / 2) / kPixels;

constexpr uint32_t kIn = kRam;              // input feature map
constexpr uint32_t kOut = kRam + 0x100000;  // pooled vector

static uint8_t buf[2][kChunkBytes];
static int16_t acc[C];
static uint8_t out[kOutBytes];

static void issue(uint32_t chunk) {
    const uint32_t off = chunk * kChunkBytes;
    const uint32_t len = kInBytes - off < kChunkBytes ? kInBytes - off : kChunkBytes;
    dma(kIn + off, addr(buf[chunk & 1]), len);
}

static void reduce(const uint8_t* src, uint32_t groups) {
    for (uint32_t cb = 0; cb < C;) {
        const size_t vl = __riscv_vsetvl_e8m4(C - cb);
        vint16m8_t a = __riscv_vle16_v_i16m8(acc + cb, vl);
        const int8_t* p = reinterpret_cast<const int8_t*>(src) + cb;
        for (uint32_t g = 0; g < groups; ++g, p += C) {
            const vint8m4_t b = __riscv_vle8_v_i8m4(p, vl);
#if INT8
            a = __riscv_vwadd_wv_i16m8(a, b, vl);
#else
            a = __riscv_vwadd_wv_i16m8(a, __riscv_vsra_vx_i8m4(__riscv_vsll_vx_i8m4(b, 4, vl), 4, vl), vl);
            a = __riscv_vwadd_wv_i16m8(a, __riscv_vsra_vx_i8m4(b, 4, vl), vl);
#endif
        }
        __riscv_vse16_v_i16m8(acc + cb, a, vl);
        cb += vl;
    }
}

// acc[from .. from+vl) -> requantized, clamped int8 (int4 range unless INT8).
static vint8m2_t requant(uint32_t from, size_t vl) {
    vint32m8_t s = __riscv_vsext_vf2_i32m8(__riscv_vle16_v_i16m4(acc + from, vl), vl);
    s = __riscv_vmul_vx_i32m8(s, kRecip, vl);
    s = __riscv_vsra_vx_i32m8(__riscv_vadd_vx_i32m8(s, 1 << (kShift - 1), vl), kShift, vl);
    const int32_t lo = INT8 ? -128 : -8, hi = INT8 ? 127 : 7;
    s = __riscv_vmin_vx_i32m8(__riscv_vmax_vx_i32m8(s, lo, vl), hi, vl);
    return __riscv_vncvt_x_x_w_i8m2(__riscv_vncvt_x_x_w_i16m4(s, vl), vl);
}

static void epilogue() {
#if INT8
    for (uint32_t j = 0; j < C;) {
        const size_t vl = __riscv_vsetvl_e32m8(C - j);
        __riscv_vse8_v_i8m2(reinterpret_cast<int8_t*>(out) + j, requant(j, vl), vl);
        j += vl;
    }
#else
    for (uint32_t j = 0; j < C / 2;) {
        const size_t vl = __riscv_vsetvl_e32m8(C / 2 - j);
        const vint8m2_t first = requant(j, vl);
        const vint8m2_t second = requant(C / 2 + j, vl);
        const vint8m2_t packed = __riscv_vor_vv_i8m2(__riscv_vand_vx_i8m2(first, 0xF, vl),
                                                     __riscv_vsll_vx_i8m2(second, 4, vl), vl);
        __riscv_vse8_v_i8m2(reinterpret_cast<int8_t*>(out) + j, packed, vl);
        j += vl;
    }
#endif
}

int main() {
    roi_begin();
    issue(0);
    for (uint32_t cb = 0; cb < C;) {
        const size_t vl = __riscv_vsetvl_e16m8(C - cb);
        __riscv_vse16_v_i16m8(acc + cb, __riscv_vmv_v_x_i16m8(0, vl), vl);
        cb += vl;
    }
    for (uint32_t i = 0; i < kChunks; ++i) {
        dma_wait();
        if (i + 1 < kChunks) {
            issue(i + 1);
        }
        const uint32_t done = i * kChunkGroups;
        reduce(buf[i & 1], kGroups - done < kChunkGroups ? kGroups - done : kChunkGroups);
    }
    mark(); // reduction done
    epilogue();
    dma(addr(out), kOut, kOutBytes);
    dma_wait();
    roi_end();
    return 0;
}
