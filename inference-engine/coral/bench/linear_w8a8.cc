// One W8A8 projection with bias, hidden -> inner (the MLP's c_fc).

#include "fill.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
float x[bench::kHidden];
std::int8_t wt[bench::kHidden * bench::kInner];
float w_scale[bench::kInner];
float bias[bench::kInner];
std::int8_t x_q[bench::kHidden];
float y[bench::kInner];
} // namespace

int main() {
    bench::Fill fill;
    fill.floats(x, bench::kHidden, 2.0F);
    fill.int8s(wt, bench::kHidden * bench::kInner);
    fill.floats(w_scale, bench::kInner, 0.001F);
    fill.floats(bias, bench::kInner, 0.1F);
    reef_kernels::linear_w8a8_bias(1, bench::kHidden, bench::kInner, x, wt,
                                   w_scale, bias, x_q, y);
    return 0;
}
