// exp, tanh and GELU against math.hpp and gelu().

#include "check.hpp"
#include "golden/math.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
float out[golden::kScratchFloats];
}

int main() {
    std::size_t which = 0;
    for (const golden::UnaryCase& c : golden::kExp) {
        reef_kernels::exp_nonpositive_f32(c.n, c.x, out);
        check::expect_bits(which++, out, c.y, c.n);
    }
    for (const golden::UnaryCase& c : golden::kTanh) {
        reef_kernels::tanh_f32(c.n, c.x, out);
        check::expect_bits(which++, out, c.y, c.n);
    }
    for (const golden::UnaryCase& c : golden::kGelu) {
        reef_kernels::gelu_f32(c.n, c.x, out);
        check::expect_bits(which++, out, c.y, c.n);
    }
    return 0;
}
