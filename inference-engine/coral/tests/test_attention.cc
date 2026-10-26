// One attention head against attend_head() and causal_visibility().

#include "check.hpp"
#include "golden/attention.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
float out[golden::kScratchFloats];
float scores[golden::kScratchScores];
} // namespace

int main() {
    std::size_t which = 0;
    for (const golden::AttentionCase& c : golden::kAttention) {
        reef_kernels::attention_head_f32(
            c.query_len, c.key_len, c.head_dim, c.query_start, c.window, c.q,
            c.head_dim, c.k, c.v, scores, out, c.head_dim);
        check::expect_bits(which++, out, c.y, c.query_len * c.head_dim);
    }
    return 0;
}
