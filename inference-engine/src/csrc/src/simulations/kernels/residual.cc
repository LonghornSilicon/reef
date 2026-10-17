// Residual add for Coral NPU: dequantize both inputs, add, requantize. See
// residual.h.
#include "residual.h"

#include "quant_helpers.h"

extern "C" void residual_add(size_t n, const int8_t* a, float a_scale,
                             const int8_t* b, float b_scale, float out_scale,
                             int8_t* out) {
    for (size_t i = 0; i < n; ++i) {
        const float value = (a_scale * static_cast<float>(a[i])) +
                            (b_scale * static_cast<float>(b[i]));
        out[i] = quant::to_int8(value, out_scale);
    }
}
