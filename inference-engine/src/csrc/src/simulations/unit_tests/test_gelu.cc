// Checks the GeLU lookup table against GPT-2's tanh GeLU in double for all
// 256 int8 inputs, and that gelu() applies it. Cycle counts come from the
// sim benches in sim/.
//
// Host build, from simulations/:
//   clang++ -std=c++17 -Ikernels -Iincludes kernels/gelu.cc \
//       unit_tests/test_gelu.cc -o test_gelu
#include "gelu.h"
#include "params.h"

#include <stddef.h>
#include <stdint.h>

extern "C" double tanh(double);
extern "C" int printf(const char*, ...);

namespace {

double gelu_reference(double x) {
    constexpr double kSqrt2OverPi = 0.7978845608028654;
    return 0.5 * x * (1.0 + tanh(kSqrt2OverPi * (x + (0.044715 * x * x * x))));
}

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
    int8_t table[256];
    int8_t x[256];
    int8_t y[256];
    gelu_table(params::kGeluInScale, params::kGeluOutScale, &table[0]);
    for (int32_t i = 0; i < 256; ++i) {
        x[i] = static_cast<int8_t>(i - 128);
    }
    gelu(256, &x[0], &table[0], &y[0]);

    // the table rounds the exact value, so it may differ by at most half a
    // step (plus clamping at the int8 range)
    double worst = 0;
    for (int32_t i = 0; i < 256; ++i) {
        const double real = static_cast<double>(x[i]) * params::kGeluInScale;
        const double exact =
            clamp_int8(gelu_reference(real) / params::kGeluOutScale);
        const double err = abs_double(y[i] - exact);
        worst = err > worst ? err : worst;
    }
    const bool pass = worst <= 0.5 + 1e-6;
    printf("gelu table     %s (max err %.3f LSB over all 256 inputs)\n",
           pass ? "pass" : "FAIL", worst);
    return pass ? 0 : 1;
}
