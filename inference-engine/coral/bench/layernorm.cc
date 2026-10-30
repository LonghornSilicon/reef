// One LayerNorm row of the hidden width.

#include "fill.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
float x[bench::kHidden];
float gain[bench::kHidden];
float shift[bench::kHidden];
float y[bench::kHidden];
} // namespace

int main() {
    bench::Fill fill;
    fill.floats(x, bench::kHidden, 3.0F);
    fill.floats(gain, bench::kHidden, 1.0F);
    fill.floats(shift, bench::kHidden, 0.5F);
    bench::roi_begin();
    reef_kernels::layernorm_f32(1, bench::kHidden, x, gain, shift, 1e-5F, y);
    bench::roi_end();
    return 0;
}
