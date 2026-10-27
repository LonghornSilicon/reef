// One float projection, hidden -> hidden (an attention q/k/v projection).

#include "fill.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
float x[bench::kHidden];
float wt[bench::kHidden * bench::kHidden];
float y[bench::kHidden];
} // namespace

int main() {
    bench::Fill fill;
    fill.floats(x, bench::kHidden, 2.0F);
    fill.floats(wt, bench::kHidden * bench::kHidden, 0.1F);
    reef_kernels::linear_f32(1, bench::kHidden, bench::kHidden, x, wt, y);
    return 0;
}
