// Matrix-engine ceiling with every operand already in DTCM (no DMA): the
// roofline's compute roof as the core can feed it. Same inner loop as
// coral/kernels/linear.cc (mac4: 4 A rows + 4x4 B rows per step, 4 vtmm into
// 4 accumulators), on a K x 64 weight block, tm = TM rows.
//   -DTM=8|16       tile rows (pixels / tokens per tile)
//   -DINT4=1        weights stored packed int4: unpack each 2 KB chunk first
//   -DNATIVE4=1     proxy for a hypothetical native-int4 Zvt (8 int4 per PE
//                   lane, k=8 per vtmm): the int8 loop on packed bytes, no
//                   unpack, half the k-steps. Real MACs = 2x reported.
#include "ops.hpp"

using namespace ops;

#ifndef TM
#define TM 16
#endif
#ifndef INT4
#define INT4 0
#endif
#ifndef NATIVE4
#define NATIVE4 0
#endif
constexpr uint32_t kK = NATIVE4 ? 256 : 512; // k-steps x 4 (native int4: K=512 packed)
constexpr uint32_t kRows = 32;      // weight rows per chunk (2 KB at int8)
constexpr uint32_t kGroups = 8;     // 64-column groups: N = 512

static int8_t act[kK * 16 + 16];
static uint8_t wbuf[kRows * 64];
#if INT4
static uint8_t packed[kRows * 32];
#endif

int main() {
    matrix_on();
    roi_begin();
    for (uint32_t g = 0; g < kGroups; ++g) {
        vtzero(true);
        for (uint32_t c = 0; c < kK / kRows; ++c) {
#if INT4
            unpack_i4_rows(packed, reinterpret_cast<int8_t*>(wbuf), kRows * 32);
#endif
            config_int8(TM);
            mac4(act + c * kRows * 16, 16, wbuf, kRows / 4);
        }
    }
    roi_end();
    return 0;
}
