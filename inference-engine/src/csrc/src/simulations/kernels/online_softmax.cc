// Online softmax for Coral NPU: one pass for the max and the normalizer,
// one pass to normalize and quantize. See online_softmax.h.
#include "online_softmax.h"

#include "quant_helpers.h"

extern "C" void online_softmax(const int32_t* scores, size_t lo, size_t hi,
                               float score_scale, uint8_t* probs) {
    // pass 1: running max m and sum l = sum exp(s - m), rescaled when m grows
    float m = score_scale * static_cast<float>(scores[lo]);
    float l = 1.0F;
    for (size_t j = lo + 1; j <= hi; ++j) {
        const float s = score_scale * static_cast<float>(scores[j]);
        if (s > m) {
            l = (l * expf(m - s)) + 1.0F;
            m = s;
        } else {
            l += expf(s - m);
        }
    }

    // pass 2: p = exp(s - m) / l, quantized to uint8 with scale 1/255
    const float to_u8 = 255.0F / l;
    for (size_t j = lo; j <= hi; ++j) {
        const float s = score_scale * static_cast<float>(scores[j]);
        // p >= 0, so adding 0.5 and truncating rounds to nearest
        // NOLINTNEXTLINE(bugprone-incorrect-roundings)
        probs[j] = static_cast<uint8_t>((expf(s - m) * to_u8) + 0.5F);
    }
}
