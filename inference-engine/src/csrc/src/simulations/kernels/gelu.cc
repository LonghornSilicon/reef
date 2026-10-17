// GeLU for Coral NPU: build a 256-entry table once, then look up. See gelu.h.
#include "gelu.h"

#include "quant_helpers.h"

extern "C" void gelu_table(float in_scale, float out_scale, int8_t* table) {
    constexpr float kSqrt2OverPi = 0.7978845608F;
    for (int32_t i = 0; i < 256; ++i) {
        const float x = static_cast<float>(i - 128) * in_scale;
        const float inner = kSqrt2OverPi * (x + (0.044715F * x * x * x));
        table[i] = quant::to_int8(0.5F * x * (1.0F + tanhf(inner)), out_scale);
    }
}

extern "C" void gelu(size_t n, const int8_t* x, const int8_t* table,
                     int8_t* y) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = table[x[i] + 128];
    }
}
