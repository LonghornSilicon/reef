// Channel concat: A[H x W x CA] ++ B[H x W x CB] -> O[H x W x (CA+CB)], int4,
// PSRAM. YOLO neck: 40x40x(256+128) and 80x80x(128+64).
//
// Layout (ops.hpp row-pair packing applied to pixels): a pixel-pair group of
// a C-channel map is C bytes (byte c = channel c of pixel 2p low nibble, of
// pixel 2p+1 high nibble). So channel offset CA always lands on byte CA of the
// output group: O group g = [A group g (CA bytes) | B group g (CB bytes)].
//
//   -DMODE=0  copy through DTCM (the measured concat op): chunks of A and B
//             groups DMA'd in, vector copy into interleaved O groups, O chunk
//             DMA'd out. Double-buffered: at chunk i the DMA writes O chunk
//             i-1 and reads A,B chunk i+1 while the vector unit copies chunk i.
//   -DMODE=1  producer stores only, contiguous: what the two producing kernels
//             pay anyway to write A and B as separate maps (the reference).
//   -DMODE=2  "virtual concat": the producers write the same data straight
//             into their channel slots of the shared O buffer (one DMA per
//             pixel-pair segment, stride CA+CB). No concat op at all;
//             MODE2 - MODE1 is its whole cost.
//   -DMODE=3  DMA-only concat, PSRAM -> PSRAM per segment, no DTCM, no core.
//   -DINT8=1  int8 (group = 1 pixel, same byte counts per group x2 groups)
#include "ops.hpp"

using namespace ops;

#ifndef H
#define H 40
#endif
#ifndef W
#define W 40
#endif
#ifndef CA
#define CA 256
#endif
#ifndef CB
#define CB 128
#endif
#ifndef MODE
#define MODE 0
#endif
#ifndef INT8
#define INT8 0
#endif

constexpr uint32_t kCT = CA + CB;
constexpr uint32_t kGroups = H * W / (INT8 ? 1 : 2); // C-byte groups per map
constexpr uint32_t kChunkGroups = 6144 / kCT;          // 6 KB of O per chunk
constexpr uint32_t kChunks = (kGroups + kChunkGroups - 1) / kChunkGroups;
constexpr uint32_t kA = kRam;
constexpr uint32_t kB = kRam + 0x100000;
constexpr uint32_t kO = kRam + 0x200000;
static_assert((H * W) % 2 == 0 && CA % 2 == 0 && CB % 2 == 0, "shape");

[[maybe_unused]] static uint32_t groups_in(uint32_t chunk) {
    const uint32_t done = chunk * kChunkGroups;
    return kGroups - done < kChunkGroups ? kGroups - done : kChunkGroups;
}

#if MODE == 0
static uint8_t abuf[2][kChunkGroups * CA];
static uint8_t bbuf[2][kChunkGroups * CB];
static uint8_t obuf[2][kChunkGroups * kCT];

static void read_chunk(uint32_t i) {
    const uint32_t g0 = i * kChunkGroups, n = groups_in(i);
    dma(kA + g0 * CA, addr(abuf[i & 1]), n * CA);
    dma(kB + g0 * CB, addr(bbuf[i & 1]), n * CB);
}
static void write_chunk(uint32_t i) {
    dma(addr(obuf[i & 1]), kO + i * kChunkGroups * kCT, groups_in(i) * kCT);
}
static void copy_bytes(const uint8_t* src, uint8_t* dst, uint32_t n) {
    for (uint32_t c = 0; c < n;) {
        const size_t vl = __riscv_vsetvl_e8m8(n - c);
        __riscv_vse8_v_u8m8(dst + c, __riscv_vle8_v_u8m8(src + c, vl), vl);
        c += vl;
    }
}
static void interleave(uint32_t i) {
    const uint8_t* a = abuf[i & 1];
    const uint8_t* b = bbuf[i & 1];
    uint8_t* o = obuf[i & 1];
    for (uint32_t g = groups_in(i); g > 0; --g, a += CA, b += CB, o += kCT) {
        copy_bytes(a, o, CA);
        copy_bytes(b, o + CA, CB);
    }
}
#elif MODE == 1 || MODE == 2
// Producer output tiles (MODE 1, 2): one chunk of A and B already computed.
static uint8_t atile[kChunkGroups * CA];
static uint8_t btile[kChunkGroups * CB];
#endif

int main() {
    roi_begin();
#if MODE == 0
    read_chunk(0);
    for (uint32_t i = 0; i < kChunks; ++i) {
        dma_wait();
        if (i > 0) {
            write_chunk(i - 1);
        }
        if (i + 1 < kChunks) {
            read_chunk(i + 1);
        }
        interleave(i);
    }
    dma_wait();
    write_chunk(kChunks - 1);
#elif MODE == 1
    for (uint32_t i = 0; i < kChunks; ++i) {
        const uint32_t g0 = i * kChunkGroups, n = groups_in(i);
        dma(addr(atile), kA + g0 * CA, n * CA);
        dma(addr(btile), kB + g0 * CB, n * CB);
    }
#elif MODE == 2
    for (uint32_t i = 0; i < kChunks; ++i) {
        const uint32_t g0 = i * kChunkGroups, n = groups_in(i);
        for (uint32_t g = 0; g < n; ++g) {
            dma(addr(atile + g * CA), kO + (g0 + g) * kCT, CA);
            dma(addr(btile + g * CB), kO + (g0 + g) * kCT + CA, CB);
        }
    }
#else
    for (uint32_t g = 0; g < kGroups; ++g) {
        dma(kA + g * CA, kO + g * kCT, CA);
        dma(kB + g * CB, kO + g * kCT + CA, CB);
    }
#endif
    dma_wait();
    roi_end();
    return 0;
}
