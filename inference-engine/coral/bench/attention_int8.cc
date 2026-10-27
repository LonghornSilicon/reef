// One causal int8 head over 16 positions of width 64 (prefill), score
// shift 10.

#include "fill.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
constexpr std::size_t kPositions = 16;
constexpr std::size_t kWidth = 64;
std::uint16_t exp_table[256];
std::int8_t q[kPositions * kWidth];
std::int8_t k[kPositions * kWidth];
std::int8_t v[kPositions * kWidth];
std::int32_t scores[kPositions];
std::uint8_t probs[kPositions];
std::int8_t out[kPositions * kWidth];
} // namespace

int main() {
    bench::Fill fill;
    fill.integers(exp_table, 256, 17);
    fill.int8s(q, kPositions * kWidth);
    fill.int8s(k, kPositions * kWidth);
    fill.int8s(v, kPositions * kWidth);
    reef_kernels::attention_head_int8(kPositions, kPositions, kWidth, 0, 0, q,
                                      kWidth, k, v, 10, exp_table, 8, scores,
                                      probs, out, kWidth);
    return 0;
}
