// Float and W8A8 linear, and the lm_head matvec, against linear().

#include "check.hpp"
#include "golden/linear.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
float out[golden::kScratchFloats];
std::int8_t x_q[golden::kScratchInt8];
} // namespace

int main() {
    std::size_t which = 0;
    for (const golden::LinearCase& c : golden::kLinear) {
        reef_kernels::linear_f32(c.rows, c.in, c.out, c.x, c.wt, out);
        check::expect_bits(which++, out, c.y, c.rows * c.out);
    }
    for (const golden::LinearBiasCase& c : golden::kLinearBias) {
        reef_kernels::linear_bias_f32(c.rows, c.in, c.out, c.x, c.wt, c.bias,
                                      out);
        check::expect_bits(which++, out, c.y, c.rows * c.out);
    }
    for (const golden::W8a8Case& c : golden::kW8a8) {
        reef_kernels::linear_w8a8(c.rows, c.in, c.out, c.x, c.wt, c.w_scale,
                                  x_q, out);
        check::expect_bits(which++, out, c.y, c.rows * c.out);
    }
    for (const golden::W8a8BiasCase& c : golden::kW8a8Bias) {
        reef_kernels::linear_w8a8_bias(c.rows, c.in, c.out, c.x, c.wt,
                                       c.w_scale, c.bias, x_q, out);
        check::expect_bits(which++, out, c.y, c.rows * c.out);
    }
    for (const golden::MatvecCase& c : golden::kMatvec) {
        reef_kernels::matvec_rows_f32(c.out, c.in, c.w, c.x, out);
        check::expect_bits(which++, out, c.y, c.out);
    }
    return 0;
}
