// Host unit tests for every kernel in kernels.h, against plain scalar or
// double-precision references. The vector unit and matrix engine are
// emulated (-DRVV_EMULATE -DVME_EMULATE), so these check logic and
// indexing; cycle counts come from the sim tests (test_*.py).
//
//   python sim.py unit
#include "kernels.h"

#include <stddef.h>
#include <stdint.h>

// host libc, declared directly so the file also parses under the
// freestanding compile_flags.txt, the same way the kernels declare expf
extern "C" double exp(double);
extern "C" double tanh(double);
extern "C" double sqrt(double);
extern "C" int printf(const char*, ...);

namespace {

// GPT-2 small quantization (bench.py Q)
constexpr float kActScale = 0.05F;
constexpr float kScoreScale = 0.05F * 0.05F / 8.0F; // q * k / sqrt(64)
constexpr float kGeluInScale = 0.05F;
constexpr float kGeluOutScale = 0.05F;
constexpr float kLnGammaScale = 0.02F;
constexpr float kLnBetaScale = 0.02F;
constexpr float kLnOutScale = 0.05F;
constexpr float kResidualOutScale = 0.05F;
constexpr double kLayernormEps = 1e-5;

// deterministic LCG so the inputs are reproducible without a library
uint32_t next_random(uint32_t* state) {
    *state = (*state * 1664525U) + 1013904223U;
    return *state >> 16;
}

int abs_int(int value) { return value < 0 ? -value : value; }

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

// ---------------------------------------------------------------------------
// Matmul: every variant against a scalar reference, including shapes that
// are not multiples of 16
// ---------------------------------------------------------------------------

constexpr size_t kMaxA = 32 * 768;
constexpr size_t kMaxB = 768 * 64;
constexpr size_t kMaxC = 32 * 64;

// too big for the stack together; main() holds one as a static local
struct Buffers {
    uint8_t a[kMaxA];
    uint8_t b[kMaxB];
    int32_t c[kMaxC];
    int32_t ref[kMaxC];
};

void fill(Buffers* buf, size_t n, size_t k, size_t m) {
    auto state = static_cast<uint32_t>((n * 131) + (k * 7) + m);
    for (size_t i = 0; i < n * k; ++i) {
        buf->a[i] = static_cast<uint8_t>(next_random(&state));
    }
    for (size_t i = 0; i < k * m; ++i) {
        buf->b[i] = static_cast<uint8_t>(next_random(&state));
    }
}

void reference(Buffers* buf, size_t n, size_t k, size_t m, bool signed_a) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            int32_t acc = 0;
            for (size_t t = 0; t < k; ++t) {
                const uint8_t raw = buf->a[(i * k) + t];
                const int32_t a = signed_a ? static_cast<int8_t>(raw) : raw;
                acc += a * static_cast<int32_t>(buf->b[(t * m) + j]);
            }
            buf->ref[(i * m) + j] = acc;
        }
    }
}

// Runs one variant on the current inputs and compares it with the reference.
template <typename Run>
int check(Buffers* buf, const char* variant, size_t n, size_t k, size_t m,
          bool signed_a, Run run) {
    for (size_t i = 0; i < n * m; ++i) {
        buf->c[i] = 0x5A5A5A5A; // poison, so unwritten outputs fail
    }
    reference(buf, n, k, m, signed_a);
    run();
    size_t bad = 0;
    for (size_t i = 0; i < n * m; ++i) {
        bad += buf->c[i] != buf->ref[i] ? 1 : 0;
    }
    printf("%-16s %3zux%3zux%3zu %s", variant, n, k, m,
           bad == 0 ? "pass\n" : "FAIL");
    if (bad != 0) {
        printf(" (%zu of %zu wrong)\n", bad, n * m);
    }
    return bad == 0 ? 0 : 1;
}

int check_shape(Buffers* buf, size_t n, size_t k, size_t m) {
    fill(buf, n, k, m);
    const uint8_t* a = &buf->a[0];
    const uint8_t* b = &buf->b[0];
    int32_t* c = &buf->c[0];
    const auto* a_s8 = reinterpret_cast<const int8_t*>(a);
    auto* c_u32 = reinterpret_cast<uint32_t*>(c);
    int failures = 0;
    failures += check(buf, "matmul_rvv", n, k, m, false,
                      [&] { matmul_rvv(n, k, m, a, b, c_u32); });
    failures += check(buf, "matmul_rvv_s8u8", n, k, m, true,
                      [&] { matmul_rvv_s8u8(n, k, m, a_s8, b, c); });
    if (n % 16 == 0 && m % 16 == 0 && k % 4 == 0) {
        failures += check(buf, "vme_matmul_u8", n, k, m, false,
                          [&] { vme_matmul_u8(n, k, m, a, b, c); });
        failures += check(buf, "vme_matmul_s8u8", n, k, m, true,
                          [&] { vme_matmul_s8u8(n, k, m, a_s8, b, c); });
    }
    failures += check(buf, "matmul (u8)", n, k, m, false,
                      [&] { matmul(n, k, m, a, false, b, c); });
    failures += check(buf, "matmul (s8)", n, k, m, true,
                      [&] { matmul(n, k, m, a, true, b, c); });
    return failures;
}

int test_matmul() {
    static Buffers buf;
    int failures = 0;
    failures += check_shape(&buf, 1, 768, 64); // decode: q . kt, x . W tile
    failures += check_shape(&buf, 7, 13, 23);  // nothing a multiple of 16
    failures += check_shape(&buf, 3, 5, 40);   // m > 16: several RVV strips
    failures += check_shape(&buf, 16, 4, 16);  // one VME step
    failures += check_shape(&buf, 32, 8, 48);
    failures += check_shape(&buf, 16, 768, 64); // prefill tile
    failures += check_shape(&buf, 32, 64, 64);
    return failures;
}

// ---------------------------------------------------------------------------
// Attention: VME block-flash attention against double-precision attention
// ---------------------------------------------------------------------------

constexpr size_t kMaxQ = 16 * 64;   // largest nq * d used below
constexpr size_t kMaxKV = 256 * 64; // largest nk * d used below
constexpr size_t kMaxKeys = 256;
constexpr size_t kBlock = 16;   // VME block when nk allows it
constexpr size_t kOddBlock = 3; // otherwise; divides none of the nk
constexpr float kAttnScoreScale = 2e-4F;
constexpr float kVScale = 0.1F;
constexpr float kAttnOutScale = 0.1F;
constexpr int32_t kKZero = 128;
constexpr int32_t kVZero = 128;

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
    const double scaled = value / kAttnOutScale;
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

        double scores[kMaxKeys];
        double m = 0;
        for (size_t j = lo; j <= pos; ++j) {
            double dot = 0;
            for (size_t c = 0; c < d; ++c) {
                dot += static_cast<double>(q[(row * d) + c]) *
                       (static_cast<double>(kt[(c * nk) + j]) - kKZero);
            }
            scores[j] = kAttnScoreScale * dot;
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

// Streams K and V block by block, the way the host does through DTCM: each
// block of kt is copied out to its own [d x len] buffer. 16-key blocks when
// nk allows, so prefill takes the matrix-engine path; otherwise 3-key
// blocks, which exercise the vector-unit fallback and ragged block edges.
void vme_attention(size_t nq, size_t nk, size_t d, size_t query_start,
                   size_t window, const int8_t* q, const uint8_t* kt,
                   const uint8_t* v, int8_t* out) {
    float m[kMaxQ];
    float l[kMaxQ];
    float acc[kMaxQ];
    uint8_t kt_block[kBlock * kMaxKV / kMaxKeys];
    // scores / P.V (nq x max(len, d) int32) then P (nq x len bytes)
    static int32_t scratch[kMaxQ + (kMaxQ / 4)];
    static uint16_t exp_table[kSoftmaxLutEntries];
    softmax_lut_table(kAttnScoreScale, &exp_table[0]);
    const size_t block = nk % kBlock == 0 ? kBlock : kOddBlock;
    for (size_t i = 0; i < nq; ++i) {
        l[i] = 0;
    }
    for (size_t i = 0; i < nq * d; ++i) {
        acc[i] = 0;
    }
    for (size_t start = 0; start < nk; start += block) {
        const size_t len = nk - start < block ? nk - start : block;
        for (size_t c = 0; c < d; ++c) {
            for (size_t j = 0; j < len; ++j) {
                kt_block[(c * len) + j] = kt[(c * nk) + start + j];
            }
        }
        vme_attention_block(nq, d, query_start, window, start, len, q,
                            &kt_block[0], kKZero, v + (start * d),
                            kAttnScoreScale, &exp_table[0], &scratch[0], &m[0],
                            &l[0], &acc[0]);
    }
    attention_flash_finish(nq, d, &l[0], &acc[0], kVZero, kVScale,
                           kAttnOutScale, out);
}

// VME attention must sit within one int8 step of the double reference
// (output rounding; the uint8 probabilities stay inside that step).
int check_attention(const char* name, size_t nq, size_t nk, size_t d,
                    size_t query_start, size_t window) {
    static int8_t q[kMaxQ];
    static uint8_t kt[kMaxKV];
    static uint8_t v[kMaxKV];
    static int8_t vme_out[kMaxQ];
    static int8_t reference_out[kMaxQ];

    fill_inputs(nq, nk, d, &q[0], &kt[0], &v[0]);
    vme_attention(nq, nk, d, query_start, window, &q[0], &kt[0], &v[0],
                  &vme_out[0]);
    reference_attention(nq, nk, d, query_start, window, &q[0], &kt[0], &v[0],
                        &reference_out[0]);
    int worst = 0;
    for (size_t i = 0; i < nq * d; ++i) {
        const int diff = abs_int(vme_out[i] - reference_out[i]);
        worst = diff > worst ? diff : worst;
    }
    const bool pass = worst <= 1;
    printf("attention %-16s %s (vs ref +-%d)\n", name, pass ? "pass" : "FAIL",
           worst);
    return pass ? 0 : 1;
}

int test_attention() {
    int failures = 0;
    // window = nk disables the local limit (global attention)
    failures += check_attention("global prefill", 6, 6, 8, 0, 6);
    failures += check_attention("global decode", 1, 7, 8, 6, 7);
    failures += check_attention("chunked decode", 3, 7, 8, 4, 7);
    failures += check_attention("local prefill", 6, 6, 8, 0, 3);
    failures += check_attention("local decode", 1, 7, 8, 6, 3);
    failures += check_attention("window of one", 1, 7, 8, 6, 1);
    // GPT-2 head width (d = 64) with full causal attention (window = nk)
    failures += check_attention("gpt2 prefill", 16, 64, 64, 48, 64);
    failures += check_attention("gpt2 decode", 1, 256, 64, 255, 256);
    // GPT-Neo 28M head width (d = 32); its local layers use a window
    failures += check_attention("neo prefill", 16, 64, 32, 48, 64);
    failures += check_attention("neo local", 16, 256, 32, 240, 64);
    failures += check_attention("neo decode", 1, 256, 32, 255, 64);
    return failures;
}

// exp_lut against exp() for every gap up to where it reads as 0: the Q15
// product of two rounded table entries stays within 2 Q15 steps.
int test_exp_lut() {
    static uint16_t table[kSoftmaxLutEntries];
    softmax_lut_table(kScoreScale, &table[0]);
    double worst = 0;
    for (uint32_t gap = 0; gap < 70000; ++gap) {
        const double exact =
            32768.0 * exp(-static_cast<double>(kScoreScale) * gap);
        const double err = abs_double(exp_lut(gap, &table[0]) - exact);
        worst = err > worst ? err : worst;
    }
    const bool pass = worst <= 2.0;
    printf("exp_lut                    %s (max err %.3f Q15 steps)\n",
           pass ? "pass" : "FAIL", worst);
    return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Pointwise
// ---------------------------------------------------------------------------

double gelu_reference(double x) {
    constexpr double kSqrt2OverPi = 0.7978845608028654;
    return 0.5 * x * (1.0 + tanh(kSqrt2OverPi * (x + (0.044715 * x * x * x))));
}

// The table rounds the exact GeLU, so it may differ by at most half a step
// (plus clamping at the int8 range), over all 256 int8 inputs.
int test_gelu() {
    int8_t table[256];
    int8_t x[256];
    int8_t y[256];
    gelu_table(kGeluInScale, kGeluOutScale, &table[0]);
    for (int32_t i = 0; i < 256; ++i) {
        x[i] = static_cast<int8_t>(i - 128);
    }
    gelu(256, &x[0], &table[0], &y[0]);

    // the table rounds the exact value, so it may differ by at most half a
    // step (plus clamping at the int8 range)
    double worst = 0;
    for (int32_t i = 0; i < 256; ++i) {
        const double real = static_cast<double>(x[i]) * kGeluInScale;
        const double exact = clamp_int8(gelu_reference(real) / kGeluOutScale);
        const double err = abs_double(y[i] - exact);
        worst = err > worst ? err : worst;
    }
    const bool pass = worst <= 0.5 + 1e-6;
    printf("gelu                       %s (max err %.3f LSB over all 256 "
           "inputs)\n",
           pass ? "pass" : "FAIL", worst);
    return pass ? 0 : 1;
}

constexpr size_t kRows = 4;
constexpr size_t kD = 768; // GPT-2 d_model

int8_t random_int8(uint32_t* state, int32_t spread) {
    return static_cast<int8_t>(
        static_cast<int32_t>(next_random(state) % ((2 * spread) + 1)) - spread);
}

int test_layernorm() {
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
    layernorm(kRows, kD, &x[0], kActScale, &gamma[0], kLnGammaScale, &beta[0],
              kLnBetaScale, kLnOutScale, &out[0]);

    double worst = 0;
    for (size_t r = 0; r < kRows; ++r) {
        double real[kD]; // dequantized row
        double mean = 0;
        double var = 0;
        for (size_t c = 0; c < kD; ++c) {
            real[c] = kActScale * static_cast<double>(x[(r * kD) + c]);
            mean += real[c] / kD;
        }
        for (const double value : real) {
            var += (value - mean) * (value - mean) / kD;
        }
        const double inv_std = 1.0 / sqrt(var + kLayernormEps);
        for (size_t c = 0; c < kD; ++c) {
            const double value = ((real[c] - mean) * inv_std * kLnGammaScale *
                                  static_cast<double>(gamma[c])) +
                                 (kLnBetaScale * static_cast<double>(beta[c]));
            const double exact = clamp_int8(value / kLnOutScale);
            const double err = abs_double(out[(r * kD) + c] - exact);
            worst = err > worst ? err : worst;
        }
    }
    // rounding contributes half a step; float stats may tip a value across
    const bool pass = worst <= 1.0;
    printf("layernorm                  %s (max err %.3f LSB, %zu rows x %zu)\n",
           pass ? "pass" : "FAIL", worst, kRows, kD);
    return pass ? 0 : 1;
}

constexpr float kBScale = 0.08F; // a different scale on the branch input

// Every pair of int8 inputs against a double-precision sum.
int test_residual() {
    static int8_t a[256 * 256];
    static int8_t b[256 * 256];
    static int8_t out[256 * 256];
    for (size_t i = 0; i < 256 * 256; ++i) {
        a[i] = static_cast<int8_t>(static_cast<int32_t>(i / 256) - 128);
        b[i] = static_cast<int8_t>(static_cast<int32_t>(i % 256) - 128);
    }
    residual_add(256 * 256, &a[0], kActScale, &b[0], kBScale, kResidualOutScale,
                 &out[0]);

    double worst = 0;
    for (size_t i = 0; i < 256 * 256; ++i) {
        const double sum = (kActScale * static_cast<double>(a[i])) +
                           (kBScale * static_cast<double>(b[i]));
        const double exact = clamp_int8(sum / kResidualOutScale);
        const double err = abs_double(out[i] - exact);
        worst = err > worst ? err : worst;
    }
    const bool pass = worst <= 0.5 + 1e-4;
    printf("residual_add               %s (max err %.3f LSB over all int8 "
           "pairs)\n",
           pass ? "pass" : "FAIL", worst);
    return pass ? 0 : 1;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_matmul();
    failures += test_attention();
    failures += test_exp_lut();
    failures += test_gelu();
    failures += test_layernorm();
    failures += test_residual();
    if (failures != 0) {
        printf("%d check(s) failed\n", failures);
    }
    return failures;
}
