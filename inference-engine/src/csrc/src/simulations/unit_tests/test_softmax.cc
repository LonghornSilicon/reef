// Checks online_softmax against a double-precision softmax: the max error of
// each uint8 probability in 1/255 steps, and how far each row's sum drifts
// from 255. Cycle counts come from the sim benches in sim/.
//
// Host build, from simulations/:
//   clang++ -std=c++17 -Ikernels -Iincludes kernels/online_softmax.cc \
//       unit_tests/test_softmax.cc -o test_softmax
#include "online_softmax.h"
#include "params.h"

#include <stddef.h>
#include <stdint.h>

extern "C" double exp(double);
extern "C" int printf(const char*, ...);

namespace {

constexpr size_t kMaxRow = 1024;

uint32_t next_random(uint32_t* state) {
    *state = (*state * 1664525U) + 1013904223U;
    return *state >> 16;
}

double abs_double(double value) { return value < 0 ? -value : value; }

// Fills scores[lo..hi] with accumulators in [-spread, spread].
void fill(int32_t* scores, size_t lo, size_t hi, int32_t spread) {
    uint32_t state = 7;
    for (size_t j = lo; j <= hi; ++j) {
        const auto r = static_cast<int32_t>(next_random(&state) % 65536);
        scores[j] = ((r - 32768) * spread) / 32768;
    }
}

// The kernel must sit within one 1/255 step of the exact probability.
int check_case(const char* name, size_t lo, size_t hi, int32_t spread) {
    int32_t scores[kMaxRow];
    uint8_t probs[kMaxRow];
    fill(&scores[0], lo, hi, spread);
    online_softmax(&scores[0], lo, hi, params::kScoreScale, &probs[0]);

    double m = params::kScoreScale * static_cast<double>(scores[lo]);
    for (size_t j = lo; j <= hi; ++j) {
        const double s = params::kScoreScale * static_cast<double>(scores[j]);
        m = s > m ? s : m;
    }
    double sum = 0;
    for (size_t j = lo; j <= hi; ++j) {
        sum += exp((params::kScoreScale * static_cast<double>(scores[j])) - m);
    }

    double worst = 0;
    int p_sum = 0;
    for (size_t j = lo; j <= hi; ++j) {
        const double exact =
            255.0 *
            exp((params::kScoreScale * static_cast<double>(scores[j])) - m) /
            sum;
        const double err = abs_double(probs[j] - exact);
        worst = err > worst ? err : worst;
        p_sum += probs[j];
    }

    const bool pass = worst <= 1.0;
    printf("%-14s %s (max err %.3f LSB, row sum %d of 255)\n", name,
           pass ? "pass" : "FAIL", worst, p_sum);
    return pass ? 0 : 1;
}

} // namespace

int main() {
    int failures = 0;
    failures += check_case("full row", 0, kMaxRow - 1, 40000);
    failures += check_case("window slice", 300, 427, 40000);
    failures += check_case("one key", 5, 5, 40000);
    failures += check_case("flat scores", 0, 63, 0);
    failures += check_case("peaked", 0, 255, 400000);
    failures += check_case("decode tail", 1000, 1023, 40000);
    if (failures != 0) {
        printf("%d case(s) failed\n", failures);
    }
    return failures;
}
