#include "../fixtures/mlp_fixture.h"
#include "../kernels/mlp_w8a8.h"
#include <math.h>
#ifdef REEF_HOST_TEST
#include <stdio.h>
#endif

// Symbols survive main() for inspection in the functional simulator.
extern "C" {
volatile int mlp_status = -1;
volatile unsigned mlp_failures = 0;
float mlp_output[kTokens * kWidth];
}
namespace {
int32_t a1[kTokens * kHidden];
int32_t a2[kTokens * kWidth];
float hidden[kTokens * kHidden];
int8_t hidden_q[kTokens * kHidden];
float hidden_scale[kTokens];
int8_t input[kTokens * kWidth];
float input_scales[kTokens];

void check(bool condition) {
    if (!condition) mlp_failures = mlp_failures + 1;
}
void close(float actual, float expected) {
    check(isfinite(actual) &&
          fabsf(actual - expected) <= 2e-6f + 2e-5f * fabsf(expected));
}
} // namespace

int main() {
    mlp_failures = 0;
    quantize_tokens(kTokens, kWidth, input_fp32, input, input_scales);
    for (unsigned i = 0; i < kTokens * kWidth; ++i)
        check(input[i] == input_q[i]);
    for (unsigned i = 0; i < kTokens; ++i)
        close(input_scales[i], input_scale[i]);
    const bool ok = mlp_w8a8(kTokens, kWidth, kHidden, input, input_scales,
                            w1, s1, b1, w2, s2, b2, a1, hidden, hidden_q,
                            hidden_scale, a2, mlp_output);
    check(ok);
    for (unsigned i = 0; i < kTokens * kHidden; ++i) {
        check(a1[i] == expected_a1[i]);
        close(hidden[i], expected_hidden[i]);
        check(hidden_q[i] == expected_hidden_q[i]);
    }
    for (unsigned i = 0; i < kTokens; ++i)
        close(hidden_scale[i], expected_hidden_scale[i]);
    for (unsigned i = 0; i < kTokens * kWidth; ++i) {
        check(a2[i] == expected_a2[i]);
        close(mlp_output[i], expected_output[i]);
    }
    // Explicit ties-to-even and zero-token checks, with scale exactly 1.
    const float ties[] = {-127, -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 127};
    const int8_t tie_expected[] = {-127, -2, -2, 0, 0, 2, 2, 127};
    int8_t tie_q[8];
    float tie_scale[1];
    quantize_tokens(1, 8, ties, tie_q, tie_scale);
    for (unsigned i = 0; i < 8; ++i) check(tie_q[i] == tie_expected[i]);
    const float zeros[8] = {};
    quantize_tokens(1, 8, zeros, tie_q, tie_scale);
    check(tie_scale[0] == 1);
    for (unsigned i = 0; i < 8; ++i) check(tie_q[i] == 0);
    check(!mlp_w8a8(kTokens, 0, kHidden, input, input_scales, w1, s1, b1,
                    w2, s2, b2, a1, hidden, hidden_q, hidden_scale, a2,
                    mlp_output));
    mlp_status = mlp_failures == 0 ? 0 : 1;
#ifdef REEF_HOST_TEST
    printf("MLP status=%d failures=%u (%u tokens, %u -> %u -> %u)\n",
           mlp_status, mlp_failures, kTokens, kWidth, kHidden, kWidth);
#endif
    return mlp_status;
}
