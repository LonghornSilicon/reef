// quant_helpers.h — scalar quantization helpers shared by the kernels.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define QUANT_INLINE static inline __attribute__((always_inline))

// newlib's math, declared directly: the freestanding build has no <math.h>.
extern "C" float expf(float);
extern "C" float tanhf(float);
extern "C" float sqrtf(float);

namespace quant {

// Round a dequantized result to the nearest int8 step and clamp to range.
QUANT_INLINE int8_t to_int8(float value, float out_scale) {
    const float scaled = value / out_scale;
    const float rounded = scaled >= 0 ? scaled + 0.5F : scaled - 0.5F;
    if (rounded >= 127.0F) {
        return 127;
    }
    if (rounded <= -128.0F) {
        return -128;
    }
    return static_cast<int8_t>(rounded);
}

} // namespace quant
