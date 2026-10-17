// Checks layernorm against a double-precision LayerNorm on random int8 rows
// of GPT-2 width. Cycle counts come from the sim benches in sim/.
//
// Host build, from simulations/:
//   clang++ -std=c++17 -Ikernels -Iincludes kernels/layernorm.cc \
//       unit_tests/test_layernorm.cc -o test_layernorm
#include "layernorm.h"
#include "params.h"

#include <stddef.h>
#include <stdint.h>

extern "C" double sqrt(double);
extern "C" int printf(const char*, ...);

namespace {

constexpr size_t kRows = 4;
constexpr size_t kD = params::kDModel;

uint32_t next_random(uint32_t* state) {
    *state = (*state * 1664525U) + 1013904223U;
    return *state >> 16;
}

int8_t random_int8(uint32_t* state, int32_t spread) {
    return static_cast<int8_t>(
        static_cast<int32_t>(next_random(state) % ((2 * spread) + 1)) - spread);
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
    static int8_t x[kRows * kD];
    static int8_t out[kRows * kD];
    int8_t gamma[kD];
    int8_t beta[kD];
    uint32_t state = 3;
    for (size_t i = 0; i < kRows * kD; ++i) {
        // row r gets spread 10 * (r + 1), so rows differ in variance
        x[i] = random_int8(&state, static_cast<int32_t>(10 * (1 + (i / kD))));
    }
    for (size_t c = 0; c < kD; ++c) {
        gamma[c] = static_cast<int8_t>(40 + random_int8(&state, 20));
        beta[c] = random_int8(&state, 30);
    }
    layernorm(kRows, kD, &x[0], params::kActScale, &gamma[0],
              params::kLnGammaScale, &beta[0], params::kLnBetaScale,
              params::kLnOutScale, &out[0]);

    double worst = 0;
    for (size_t r = 0; r < kRows; ++r) {
        double real[kD]; // dequantized row
        double mean = 0;
        double var = 0;
        for (size_t c = 0; c < kD; ++c) {
            real[c] = params::kActScale * static_cast<double>(x[(r * kD) + c]);
            mean += real[c] / kD;
        }
        for (const double value : real) {
            var += (value - mean) * (value - mean) / kD;
        }
        const double inv_std = 1.0 / sqrt(var + params::kLayernormEps);
        for (size_t c = 0; c < kD; ++c) {
            const double value =
                ((real[c] - mean) * inv_std * params::kLnGammaScale *
                 static_cast<double>(gamma[c])) +
                (params::kLnBetaScale * static_cast<double>(beta[c]));
            const double exact = clamp_int8(value / params::kLnOutScale);
            const double err = abs_double(out[(r * kD) + c] - exact);
            worst = err > worst ? err : worst;
        }
    }
    // rounding contributes half a step; float stats may tip a value across
    const bool pass = worst <= 1.0;
    printf("layernorm      %s (max err %.3f LSB, %zu rows x %zu)\n",
           pass ? "pass" : "FAIL", worst, kRows, kD);
    return pass ? 0 : 1;
}
