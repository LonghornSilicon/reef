// LayerNorm against layernorm().

#include "check.hpp"
#include "golden/layernorm.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
float out[golden::kScratchFloats];
}

int main() {
    std::size_t which = 0;
    for (const golden::LayerNormCase& c : golden::kLayerNorm) {
        reef_kernels::layernorm_f32(c.rows, c.width, c.x, c.gain, c.shift,
                                    c.eps, out);
        check::expect_bits(which++, out, c.y, c.rows * c.width);
    }
    return 0;
}
