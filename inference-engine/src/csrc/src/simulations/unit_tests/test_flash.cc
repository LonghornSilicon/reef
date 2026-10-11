// Runs flash attention and serial attention on the host and checks that
// they agree with each other and with a double-precision reference.
// Cycle measurement (rdcycle) waits on the Coral simulator build; see the
// README's known issues.
//
// Host build, from simulations/:
//   clang++ -std=c++17 -Ikernels -Iincludes kernels/attention_serial.cc \
//       kernels/attention_flash.cc unit_tests/test_flash.cc -o test_flash
#include "attention_flash.h"
#include "attention_serial.h"

#include <stddef.h>
#include <stdint.h>

// host libc, declared directly so the file also parses under the
// freestanding compile_flags.txt, the same way the kernels declare expf
extern "C" double exp(double);
extern "C" int printf(const char*, ...);

namespace {

constexpr size_t kMaxSize = 16; // largest nq, nk or d used below
constexpr float kScoreScale = 2e-4F;
constexpr float kVScale = 0.1F;
constexpr float kOutScale = 0.1F;
constexpr int32_t kKZero = 128;
constexpr int32_t kVZero = 128;

// deterministic LCG so the inputs are reproducible without a library
uint32_t next_random(uint32_t* state) {
    *state = (*state * 1664525U) + 1013904223U;
    return *state >> 16;
}

int abs_int(int value) { return value < 0 ? -value : value; }

void fill_inputs(size_t nq, size_t nk, size_t d, int8_t* q, uint8_t* kt,
                 uint8_t* v) {
    uint32_t state = 1;
    for (size_t i = 0; i < nq * d; ++i) {
        const auto raw = static_cast<int32_t>(next_random(&state) % 256);
        q[i] = static_cast<int8_t>(raw - 128);
    }
    for (size_t i = 0; i < d * nk; ++i) {
        kt[i] = static_cast<uint8_t>(next_random(&state) % 256);
    }
    for (size_t i = 0; i < nk * d; ++i) {
        v[i] = static_cast<uint8_t>(next_random(&state) % 256);
    }
}

// the kernels' rounding rule (nearest, ties away from zero), in double
int8_t quantize_reference(double value) {
    const double scaled = value / kOutScale;
    const double rounded = scaled >= 0 ? scaled + 0.5 : scaled - 0.5;
    if (rounded >= 127.0) {
        return 127;
    }
    if (rounded <= -128.0) {
        return -128;
    }
    return static_cast<int8_t>(rounded);
}

// textbook attention in double precision over the dequantized inputs
void reference_attention(size_t nq, size_t nk, size_t d, size_t query_start,
                         size_t window, const int8_t* q, const uint8_t* kt,
                         const uint8_t* v, int8_t* out) {
    for (size_t row = 0; row < nq; ++row) {
        const size_t pos = query_start + row;
        const size_t lo = (pos + 1 > window) ? pos + 1 - window : 0;

        double scores[kMaxSize];
        double m = 0;
        for (size_t j = lo; j <= pos; ++j) {
            double dot = 0;
            for (size_t c = 0; c < d; ++c) {
                dot += static_cast<double>(q[(row * d) + c]) *
                       (static_cast<double>(kt[(c * nk) + j]) - kKZero);
            }
            scores[j] = kScoreScale * dot;
            if (j == lo || scores[j] > m) {
                m = scores[j];
            }
        }

        double sum = 0;
        for (size_t j = lo; j <= pos; ++j) {
            scores[j] = exp(scores[j] - m);
            sum += scores[j];
        }

        for (size_t c = 0; c < d; ++c) {
            double mixed = 0;
            for (size_t j = lo; j <= pos; ++j) {
                mixed += (scores[j] / sum) *
                         (static_cast<double>(v[(j * d) + c]) - kVZero);
            }
            out[(row * d) + c] = quantize_reference(kVScale * mixed);
        }
    }
}

// Runs one shape through all three implementations. Serial must sit within
// one int8 step of the reference (quantization rounding), and flash within
// one step of serial (float reordering).
int check_case(const char* name, size_t nq, size_t nk, size_t d,
               size_t query_start, size_t window) {
    int8_t q[kMaxSize * kMaxSize];
    uint8_t kt[kMaxSize * kMaxSize];
    uint8_t v[kMaxSize * kMaxSize];
    int8_t serial_out[kMaxSize * kMaxSize];
    int8_t flash_out[kMaxSize * kMaxSize];
    int8_t reference_out[kMaxSize * kMaxSize];

    fill_inputs(nq, nk, d, &q[0], &kt[0], &v[0]);
    attention_serial(nq, nk, d, query_start, window, &q[0], &kt[0], kKZero,
                     &v[0], kVZero, kScoreScale, kVScale, kOutScale,
                     &serial_out[0]);
    attention_flash(nq, nk, d, query_start, window, &q[0], &kt[0], kKZero,
                    &v[0], kVZero, kScoreScale, kVScale, kOutScale,
                    &flash_out[0]);
    reference_attention(nq, nk, d, query_start, window, &q[0], &kt[0], &v[0],
                        &reference_out[0]);

    int worst_serial = 0;
    int worst_flash = 0;
    for (size_t i = 0; i < nq * d; ++i) {
        const int serial_diff = abs_int(serial_out[i] - reference_out[i]);
        const int flash_diff = abs_int(flash_out[i] - serial_out[i]);
        worst_serial = serial_diff > worst_serial ? serial_diff : worst_serial;
        worst_flash = flash_diff > worst_flash ? flash_diff : worst_flash;
    }

    const bool pass = worst_serial <= 1 && worst_flash <= 1;
    printf("%-16s %s (serial vs ref +-%d, flash vs serial +-%d)\n", name,
           pass ? "pass" : "FAIL", worst_serial, worst_flash);
    return pass ? 0 : 1;
}

} // namespace

int main() {
    int failures = 0;
    // window = nk disables the local limit (global attention)
    failures += check_case("global prefill", 6, 6, 8, 0, 6);
    failures += check_case("global decode", 1, 7, 8, 6, 7);
    failures += check_case("chunked decode", 3, 7, 8, 4, 7);
    failures += check_case("local prefill", 6, 6, 8, 0, 3);
    failures += check_case("local decode", 1, 7, 8, 6, 3);
    failures += check_case("window of one", 1, 7, 8, 6, 1);
    if (failures != 0) {
        printf("%d case(s) failed\n", failures);
    }
    return failures;
}
