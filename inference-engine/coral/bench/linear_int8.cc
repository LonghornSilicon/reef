// int8 projection with bias: 16 rows, 64 -> 64, shift 8.

#include "fill.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
constexpr std::size_t kRows = 16;
constexpr std::size_t kIn = 64;
constexpr std::size_t kOut = 64;
std::int8_t x[kRows * kIn];
std::int8_t wt[kIn * kOut];
std::int32_t bias[kOut];
std::int8_t y[kRows * kOut];
} // namespace

int main() {
    bench::Fill fill;
    fill.int8s(x, kRows * kIn);
    fill.int8s(wt, kIn * kOut);
    fill.integers(bias, kOut, 20);
    reef_kernels::linear_bias_int8(kRows, kIn, kOut, x, wt, bias, 8, y);
    return 0;
}
