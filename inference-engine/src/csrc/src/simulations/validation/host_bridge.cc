// Host-only access to the unchanged scalar W8A8 kernel. No Coral dependency.
#include "../kernels/mlp_w8a8.h"

#include <cfenv>

extern "C" int
reef_validate_mlp(size_t tokens, size_t width, size_t hidden,
                  const float* input, const int8_t* w1, const float* s1,
                  const float* b1, const int8_t* w2, const float* s2,
                  const float* b2, int8_t* input_q, float* input_scale,
                  int32_t* a1, float* pre, float* hidden_fp32, int8_t* hidden_q,
                  float* hidden_scale, int32_t* a2, float* output) {
    if (std::fegetround() != FE_TONEAREST) {
        return 3;
    }
    quantize_tokens(tokens, width, input, input_q, input_scale);
    // Preserve pre-GELU values for inspection using the same Linear entry
    // point.
    if (!linear_w8a8(tokens, width, hidden, input_q, input_scale, w1, s1, b1,
                     a1, pre)) {
        return 1;
    }
    return mlp_w8a8(tokens, width, hidden, input_q, input_scale, w1, s1, b1, w2,
                    s2, b2, a1, hidden_fp32, hidden_q, hidden_scale, a2, output)
               ? 0
               : 2;
}
