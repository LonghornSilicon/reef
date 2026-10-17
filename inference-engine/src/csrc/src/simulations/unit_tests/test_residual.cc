// Checks residual_add against a double-precision reference over every pair
// of int8 inputs. Cycle counts come from the sim benches in sim/.
//
// Host build, from simulations/:
//   clang++ -std=c++17 -Ikernels -Iincludes kernels/residual.cc \
//       unit_tests/test_residual.cc -o test_residual
#include "params.h"
#include "residual.h"

#include <stddef.h>
#include <stdint.h>

extern "C" int printf(const char*, ...);

namespace {

constexpr float kBScale = 0.08F; // a different scale on the branch input

double abs_double(double value) { return value < 0 ? -value : value; }

double clamp_int8(double value) {
    if (value > 127.0) {
        return 127.0;
    }
    if (value < -128.0) {
        return -128.0;
    }
    return value;
}

} // namespace

int main() {
    static int8_t a[256 * 256];
    static int8_t b[256 * 256];
    static int8_t out[256 * 256];
    for (size_t i = 0; i < 256 * 256; ++i) {
        a[i] = static_cast<int8_t>(static_cast<int32_t>(i / 256) - 128);
        b[i] = static_cast<int8_t>(static_cast<int32_t>(i % 256) - 128);
    }
    residual_add(256 * 256, &a[0], params::kActScale, &b[0], kBScale,
                 params::kResidualOutScale, &out[0]);

    double worst = 0;
    for (size_t i = 0; i < 256 * 256; ++i) {
        const double sum = (params::kActScale * static_cast<double>(a[i])) +
                           (kBScale * static_cast<double>(b[i]));
        const double exact = clamp_int8(sum / params::kResidualOutScale);
        const double err = abs_double(out[i] - exact);
        worst = err > worst ? err : worst;
    }
    const bool pass = worst <= 0.5 + 1e-4;
    printf("residual add   %s (max err %.3f LSB over all int8 pairs)\n",
           pass ? "pass" : "FAIL", worst);
    return pass ? 0 : 1;
}
