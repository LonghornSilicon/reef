// Runs the RVV embedding kernels against a scalar reference on the Coral sim
//
// No C library exist on the core, so the result is written to memory for the
// simulator shell to read:
//   status = 1 if every element matches the scalar reference, else 0
//   checksum = sum of the RVV output (8960 for the shapes below)
//
// Per-kernel instruction counts: break on the extern "C" symbols and read
// minstret at each stop (markers are kept out of line so they stay visible)
//   break embedding_lookup_rvv / break embedding_add_position_rvv /
//   break marker_rvv_done / break marker_scalar_done
#include <stddef.h>
#include <stdint.h>

#include "embedding_rvv.h"

namespace {

constexpr size_t kVocab = 64;
constexpr size_t kWidth = 64;
constexpr size_t kMaxPositions = 32;
constexpr size_t kLength = 4;
constexpr size_t kOffset = 2;

// Test buffers live in DTCM so the simulator shell can read them by name.
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
int8_t g_table[kVocab * kWidth];
int8_t g_positions[kMaxPositions * kWidth];
uint32_t g_ids[kLength] = {3, 0, 3, 5};
int8_t g_out_rvv[kLength * kWidth];
int8_t g_out_ref[kLength * kWidth];
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

void fill_tables() {
    for (size_t v = 0; v < kVocab; v++) {
        for (size_t c = 0; c < kWidth; c++) {
            g_table[(v * kWidth) + c] = static_cast<int8_t>((v + c) & 63);
        }
    }
    for (size_t p = 0; p < kMaxPositions; p++) {
        for (size_t c = 0; c < kWidth; c++) {
            g_positions[(p * kWidth) + c] = static_cast<int8_t>(p & 31);
        }
    }
}

// Scalar reference: the inference_engine loops, minus Tensor and exceptions
void lookup_scalar(const uint32_t* ids, const int8_t* table, int8_t* out) {
    for (size_t i = 0; i < kLength; i++) {
        for (size_t c = 0; c < kWidth; c++) {
            out[(i * kWidth) + c] = table[(ids[i] * kWidth) + c];
        }
    }
}

void add_position_scalar(const int8_t* positions, int8_t* out) {
    for (size_t i = 0; i < kLength; i++) {
        for (size_t c = 0; c < kWidth; c++) {
            out[(i * kWidth) + c] =
                static_cast<int8_t>(out[(i * kWidth) + c] +
                                    positions[((kOffset + i) * kWidth) + c]);
        }
    }
}

} // namespace

// Results the simulator shell reads after the run (mem get status x32)
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
// NOLINTBEGIN(misc-use-internal-linkage)
volatile int32_t status;
volatile int32_t checksum;
// NOLINTEND(misc-use-internal-linkage)
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

extern "C" __attribute__((noinline)) void marker_rvv_done() {
    asm volatile("");
}
extern "C" __attribute__((noinline)) void marker_scalar_done() {
    asm volatile("");
}

int main() { // NOLINT(misc-use-internal-linkage)
    fill_tables();

    embedding_lookup_rvv(kLength, kWidth, &g_ids[0], &g_table[0],
                         &g_out_rvv[0]);
    embedding_add_position_rvv(kLength, kWidth, kOffset, &g_positions[0],
                               &g_out_rvv[0]);
    marker_rvv_done();

    lookup_scalar(&g_ids[0], &g_table[0], &g_out_ref[0]);
    add_position_scalar(&g_positions[0], &g_out_ref[0]);
    marker_scalar_done();

    int32_t ok = 1;
    int32_t sum = 0;
    for (size_t i = 0; i < kLength * kWidth; i++) {
        if (g_out_rvv[i] != g_out_ref[i]) {
            ok = 0;
        }
        sum += g_out_rvv[i];
    }
    status = ok;
    checksum = sum;
    return 0;
}