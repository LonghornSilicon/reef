// The tied lm_head over 256 vocabulary rows (the real 50257 rows do not fit
// in DTCM; cost scales linearly with the row count), then argmax.

#include "fill.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
constexpr std::size_t kRows = 256;
float x[bench::kHidden];
float wte[kRows * bench::kHidden];
float logits[kRows];
volatile std::uint32_t next_token;
} // namespace

int main() {
    bench::Fill fill;
    fill.floats(x, bench::kHidden, 2.0F);
    fill.floats(wte, kRows * bench::kHidden, 0.1F);
    bench::roi_begin();
    reef_kernels::matvec_rows_f32(kRows, bench::kHidden, wte, x, logits);
    next_token = reef_kernels::argmax_f32(kRows, logits);
    bench::roi_end();
    return 0;
}
