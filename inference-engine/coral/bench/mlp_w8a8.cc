// The MLP block in W8A8: c_fc (hidden -> inner), GELU, c_proj (inner ->
// hidden), as the firmware runs it.

#include "fill.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
float x[bench::kHidden];
std::int8_t fc_wt[bench::kHidden * bench::kInner];
float fc_scale[bench::kInner];
float fc_bias[bench::kInner];
std::int8_t proj_wt[bench::kInner * bench::kHidden];
float proj_scale[bench::kHidden];
float proj_bias[bench::kHidden];
std::int8_t x_q[bench::kInner];
float inner[bench::kInner];
float y[bench::kHidden];
} // namespace

int main() {
    bench::Fill fill;
    fill.floats(x, bench::kHidden, 2.0F);
    fill.int8s(fc_wt, bench::kHidden * bench::kInner);
    fill.floats(fc_scale, bench::kInner, 0.001F);
    fill.floats(fc_bias, bench::kInner, 0.1F);
    fill.int8s(proj_wt, bench::kInner * bench::kHidden);
    fill.floats(proj_scale, bench::kHidden, 0.001F);
    fill.floats(proj_bias, bench::kHidden, 0.1F);
    bench::roi_begin();
    reef_kernels::linear_w8a8_bias(1, bench::kHidden, bench::kInner, x, fc_wt,
                                   fc_scale, fc_bias, x_q, inner);
    reef_kernels::gelu_f32(bench::kInner, inner, inner);
    reef_kernels::linear_w8a8_bias(1, bench::kInner, bench::kHidden, inner,
                                   proj_wt, proj_scale, proj_bias, x_q, y);
    bench::roi_end();
    return 0;
}
