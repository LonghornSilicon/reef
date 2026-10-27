// int8 LayerNorm over 16 rows of 64.

#include "fill.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
constexpr std::size_t kRows = 16;
constexpr std::size_t kWidth = 64;
std::int8_t x[kRows * kWidth];
std::int8_t gain[kWidth];
std::int8_t shift[kWidth];
std::int8_t y[kRows * kWidth];
} // namespace

int main() {
    bench::Fill fill;
    fill.int8s(x, kRows * kWidth);
    fill.integers(gain, kWidth, 27); // -16..15; +32 below: near 1.0 in Q5
    for (std::int8_t& g : gain) {
        g = static_cast<std::int8_t>(g + 32);
    }
    fill.integers(shift, kWidth, 27);
    reef_kernels::layernorm_int8(kRows, kWidth, x, gain, shift, y);
    return 0;
}
