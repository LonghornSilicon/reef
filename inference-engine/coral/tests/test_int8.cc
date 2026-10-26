// The integer-only kernels against int8.hpp: linear, GELU table, softmax,
// attention, LayerNorm, add and embedding.

#include "check.hpp"
#include "golden/int8.hpp"
#include "reef_kernels/kernels.hpp"

#include <cstdint>

namespace {
std::int8_t out[golden::kScratchInt8];
std::int32_t scores[golden::kScratchScores];
std::uint8_t probs[golden::kScratchScores];

void expect_bytes(std::size_t which, const void* got, const void* want,
                  std::size_t n) {
    const auto* a = static_cast<const std::uint8_t*>(got);
    const auto* b = static_cast<const std::uint8_t*>(want);
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            check::fail(which, i);
        }
    }
}
} // namespace

int main() {
    std::size_t which = 0;
    for (const golden::Int8LinearCase& c : golden::kInt8Linear) {
        reef_kernels::linear_int8(c.rows, c.in, c.out, c.x, c.wt, c.shift, out);
        expect_bytes(which++, out, c.y, c.rows * c.out);
    }
    for (const golden::Int8LinearBiasCase& c : golden::kInt8LinearBias) {
        reef_kernels::linear_bias_int8(c.rows, c.in, c.out, c.x, c.wt, c.bias,
                                       c.shift, out);
        expect_bytes(which++, out, c.y, c.rows * c.out);
    }
    for (const golden::Int8TableCase& c : golden::kInt8Gelu) {
        reef_kernels::gelu_int8(c.n, c.x, c.table, out);
        expect_bytes(which++, out, c.y, c.n);
    }
    for (const golden::Int8SoftmaxCase& c : golden::kInt8Softmax) {
        for (std::size_t r = 0; r < c.rows; ++r) {
            reef_kernels::softmax_int8(c.n, c.scores + (r * c.n), c.shift,
                                       c.table, probs + (r * c.n));
        }
        expect_bytes(which++, probs, c.probs, c.rows * c.n);
    }
    for (const golden::Int8AttentionCase& c : golden::kInt8Attention) {
        reef_kernels::attention_head_int8(
            c.query_len, c.key_len, c.head_dim, c.query_start, c.window, c.q,
            c.head_dim, c.k, c.v, c.score_shift, c.table, c.out_shift, scores,
            probs, out, c.head_dim);
        expect_bytes(which++, out, c.y, c.query_len * c.head_dim);
    }
    for (const golden::Int8LayerNormCase& c : golden::kInt8LayerNorm) {
        reef_kernels::layernorm_int8(c.rows, c.width, c.x, c.gain, c.shift,
                                     out);
        expect_bytes(which++, out, c.y, c.rows * c.width);
    }
    for (const golden::Int8AddCase& c : golden::kInt8Add) {
        reef_kernels::add_int8(c.n, c.a, c.b, out);
        expect_bytes(which++, out, c.y, c.n);
    }
    for (const golden::Int8EmbeddingCase& c : golden::kInt8Embedding) {
        reef_kernels::embedding_int8(c.width, c.id, c.position, c.wte, c.wpe,
                                     out);
        expect_bytes(which++, out, c.y, c.width);
    }
    return 0;
}
