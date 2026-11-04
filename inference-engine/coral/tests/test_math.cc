// exp, tanh and GELU against math.hpp and gelu().

#include "check.hpp"
#include "golden/math.hpp"
#include "reef_kernels/kernels.hpp"
#include "vector_math.hpp"

#include <riscv_vector.h>

namespace {
float out[golden::kScratchFloats];

// exp_nonpositive() and tanh_float() elementwise, through the vector code the
// kernels inline. They live here, not in the library, to keep them out of
// the ITCM image.
void exp_nonpositive_f32(std::size_t n, const float* x, float* y) {
    for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m4(n - i);
        __riscv_vse32_v_f32m4(y + i,
                              reef_kernels::detail::exp_nonpositive(
                                  __riscv_vle32_v_f32m4(x + i, vl), vl),
                              vl);
        i += vl;
    }
}

void tanh_f32(std::size_t n, const float* x, float* y) {
    for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m4(n - i);
        __riscv_vse32_v_f32m4(
            y + i,
            reef_kernels::detail::tanh(__riscv_vle32_v_f32m4(x + i, vl), vl),
            vl);
        i += vl;
    }
}
} // namespace

int main() {
    std::size_t which = 0;
    for (const golden::UnaryCase& c : golden::kExp) {
        exp_nonpositive_f32(c.n, c.x, out);
        check::expect_bits(which++, out, c.y, c.n);
    }
    for (const golden::UnaryCase& c : golden::kTanh) {
        tanh_f32(c.n, c.x, out);
        check::expect_bits(which++, out, c.y, c.n);
    }
    for (const golden::UnaryCase& c : golden::kGelu) {
        reef_kernels::gelu_f32(c.n, c.x, out);
        check::expect_bits(which++, out, c.y, c.n);
    }
    return 0;
}
