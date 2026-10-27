// int8 GELU over 1024 values, then softmax over 16 rows of 64 scores.

#include "fill.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
constexpr std::size_t kValues = 1024;
constexpr std::size_t kRows = 16;
constexpr std::size_t kCols = 64;
std::int8_t gelu_table[256];
std::uint16_t exp_table[256];
std::int8_t x[kValues];
std::int8_t y[kValues];
std::int32_t scores[kRows * kCols];
std::uint8_t probs[kRows * kCols];
} // namespace

int main() {
    bench::Fill fill;
    fill.int8s(gelu_table, 256);
    fill.integers(exp_table, 256, 17);
    fill.int8s(x, kValues);
    fill.integers(scores, kRows * kCols, 20);
    reef_kernels::gelu_int8(kValues, x, gelu_table, y);
    for (std::size_t r = 0; r < kRows; ++r) {
        reef_kernels::softmax_int8(kCols, scores + (r * kCols), 4, exp_table,
                                   probs + (r * kCols));
    }
    return 0;
}
