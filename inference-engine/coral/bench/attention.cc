// One attention head decoding the 129th token: a query against 128 cached
// keys and values plus its own.

#include "fill.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
constexpr std::size_t kKeys = 129;
float q[bench::kHeadDim];
float k[kKeys * bench::kHeadDim];
float v[kKeys * bench::kHeadDim];
float scores[kKeys];
float out[bench::kHeadDim];
} // namespace

int main() {
    bench::Fill fill;
    fill.floats(q, bench::kHeadDim, 1.0F);
    fill.floats(k, kKeys * bench::kHeadDim, 1.0F);
    fill.floats(v, kKeys * bench::kHeadDim, 1.0F);
    reef_kernels::attention_head_f32(1, kKeys, bench::kHeadDim, kKeys - 1, 0, q,
                                     bench::kHeadDim, k, v, scores, out,
                                     bench::kHeadDim);
    return 0;
}
