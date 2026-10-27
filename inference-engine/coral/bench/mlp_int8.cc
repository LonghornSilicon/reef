// The int8 MLP block: 16 rows, c_fc 32 -> 128, GELU table, c_proj 128 ->
// 32, shift 8 for both projections.

#include "fill.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
constexpr std::size_t kRows = 16;
constexpr std::size_t kWidth = 32;
constexpr std::size_t kInner = 128;
std::int8_t gelu_table[256];
std::int8_t x[kRows * kWidth];
std::int8_t fc_wt[kWidth * kInner];
std::int32_t fc_bias[kInner];
std::int8_t proj_wt[kInner * kWidth];
std::int32_t proj_bias[kWidth];
std::int8_t inner[kRows * kInner];
std::int8_t y[kRows * kWidth];
} // namespace

int main() {
    bench::Fill fill;
    fill.int8s(gelu_table, 256);
    fill.int8s(x, kRows * kWidth);
    fill.int8s(fc_wt, kWidth * kInner);
    fill.integers(fc_bias, kInner, 20);
    fill.int8s(proj_wt, kInner * kWidth);
    fill.integers(proj_bias, kWidth, 20);
    reef_kernels::linear_bias_int8(kRows, kWidth, kInner, x, fc_wt, fc_bias, 8,
                                   inner);
    reef_kernels::gelu_int8(kRows * kInner, inner, gelu_table, inner);
    reef_kernels::linear_bias_int8(kRows, kInner, kWidth, inner, proj_wt,
                                   proj_bias, 8, y);
    return 0;
}
