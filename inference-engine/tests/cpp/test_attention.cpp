// Attention unit tests: hand-checkable math and prefill-vs-decode
// consistency, against the structs attention.hpp defines:
//   AttentionWeights { num_heads, query_weights, key_weights, value_weights,
//                      output_weights, output_bias }
//   Qkv              { query, key, value }   (batch, heads, len, head_dim)
//   KvCache          { capacity, key, value } default-constructed == empty,
//                                             capacity 0 == grow on demand
#include "inference_engine/operator/attention.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace inference_engine {

namespace {

constexpr float kTight = 1e-6F;

Tensor<float> make_tensor(const std::vector<std::size_t>& shape,
                          std::vector<float> values) {
    return Tensor<float>{shape, std::move(values)};
}

/// n-by-n identity matrix.
Tensor<float> identity(std::size_t n) {
    std::vector<float> values(n * n, 0.0F);
    for (std::size_t i = 0; i < n; ++i) {
        values[(i * n) + i] = 1.0F;
    }
    return make_tensor({n, n}, std::move(values));
}

/// Deterministic non-symmetric matrix with small values in [-0.2, 0.2],
/// so attention scores stay out of softmax saturation. Non-identity
/// weights matter for the prefill-vs-decode test: identity is its own
/// transpose and can hide layout bugs.
Tensor<float> pattern_matrix(std::size_t n, std::size_t offset) {
    std::vector<float> values(n * n);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = (0.1F * static_cast<float>(((i * 7) + offset) % 5)) - 0.2F;
    }
    return make_tensor({n, n}, std::move(values));
}

Tensor<float> zero_bias(std::size_t n) {
    return make_tensor({n}, std::vector<float>(n, 0.0F));
}

/// Weights that make forward a pure attention mix: q = k = v = input.
AttentionWeights<float> identity_weights(std::size_t hidden,
                                         std::size_t num_heads) {
    return AttentionWeights<float>{num_heads,        identity(hidden),
                                   identity(hidden), identity(hidden),
                                   identity(hidden), zero_bias(hidden)};
}

AttentionWeights<float> pattern_weights(std::size_t hidden,
                                        std::size_t num_heads) {
    return AttentionWeights<float>{num_heads,
                                   pattern_matrix(hidden, 0),
                                   pattern_matrix(hidden, 1),
                                   pattern_matrix(hidden, 2),
                                   pattern_matrix(hidden, 3),
                                   zero_bias(hidden)};
}

/// Input shaped (1, len, hidden) with small deterministic values.
Tensor<float> make_input(std::size_t len, std::size_t hidden) {
    std::vector<float> values(len * hidden);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = (0.1F * static_cast<float>(i % 7)) - 0.3F;
    }
    return make_tensor({1, len, hidden}, std::move(values));
}

/// One token row of a (1, len, hidden) output.
std::vector<float> row(const Tensor<float>& output, std::size_t index) {
    const std::size_t hidden = output.shape[2];
    const auto begin =
        output.values.begin() + static_cast<std::ptrdiff_t>(index * hidden);
    return {begin, begin + static_cast<std::ptrdiff_t>(hidden)};
}

void expect_near(const std::vector<float>& actual,
                 const std::vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "at index " << i;
    }
}

bool rows_differ(const std::vector<float>& left,
                 const std::vector<float>& right) {
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (std::fabs(left[i] - right[i]) > kTight) {
            return true;
        }
    }
    return false;
}

// With one token there is nothing to attend to but itself: the score
// matrix is 1x1, softmax yields exactly 1, and identity projections make
// the whole layer a passthrough. Catches structural bugs (projection,
// head split/merge, output reshape) before any attention math is tested.
TEST(AttentionTest, SingleTokenPassthrough) {
    const AttentionWeights<float> weights = identity_weights(2, 1);
    KvCache<float> cache;
    GlobalAttention<float> attention(weights, cache);

    const Tensor<float> input = make_tensor({1, 1, 2}, {0.3F, -0.7F});
    const Tensor<float> output = attention.forward(input);

    ASSERT_EQ(output.shape, input.shape);
    expect_near(output.values, input.values, kTight);
}

// Two tokens, one head, identity weights, so q = k = v = input.
//   t0 = [1, 0], t1 = [0, 1]
// Row 0 sees only t0 (causal): output = v0 = [1, 0].
// Row 1 scores (unscaled dot products): [t1.t0, t1.t1] = [0, 1];
// softmax([0, 1]) = [1, e] / (1 + e) = [0.26894142, 0.73105858];
// output = 0.26894142 * v0 + 0.73105858 * v1.
TEST(AttentionTest, HandComputedPrefill) {
    const AttentionWeights<float> weights = identity_weights(2, 1);
    KvCache<float> cache;
    GlobalAttention<float> attention(weights, cache);

    const Tensor<float> input =
        make_tensor({1, 2, 2}, {1.0F, 0.0F, 0.0F, 1.0F});
    const Tensor<float> output = attention.forward(input);

    ASSERT_EQ(output.shape, input.shape);
    expect_near(row(output, 0), {1.0F, 0.0F}, kTight);
    expect_near(row(output, 1), {0.26894142F, 0.73105858F}, kTight);
}

// Token 0 must be blind to token 1: change the second token and the first
// output row must not move at all. Tests the causal mask's effect without
// assuming anything about its implementation.
TEST(AttentionTest, CausalityHolds) {
    const AttentionWeights<float> weights = identity_weights(2, 1);

    KvCache<float> cache_a;
    GlobalAttention<float> attention_a(weights, cache_a);
    const Tensor<float> output_a =
        attention_a.forward(make_tensor({1, 2, 2}, {0.5F, -0.1F, 0.8F, 0.2F}));

    KvCache<float> cache_b;
    GlobalAttention<float> attention_b(weights, cache_b);
    const Tensor<float> output_b =
        attention_b.forward(make_tensor({1, 2, 2}, {0.5F, -0.1F, -0.9F, 0.4F}));

    expect_near(row(output_a, 0), row(output_b, 0), kTight);
}

// Output shape mirrors input shape, and the cache accumulates every token
// seen across a prefill and subsequent decode steps.
TEST(AttentionTest, ShapesAndCacheGrowth) {
    const std::size_t hidden = 4;
    const std::size_t heads = 2;
    const AttentionWeights<float> weights = pattern_weights(hidden, heads);
    KvCache<float> cache;
    GlobalAttention<float> attention(weights, cache);

    const Tensor<float> prefill_out = attention.forward(make_input(3, hidden));
    EXPECT_EQ(prefill_out.shape, (std::vector<std::size_t>{1, 3, hidden}));

    for (std::size_t step = 0; step < 2; ++step) {
        const Tensor<float> decode_out =
            attention.forward(make_input(1, hidden));
        EXPECT_EQ(decode_out.shape, (std::vector<std::size_t>{1, 1, hidden}));
    }

    const std::size_t head_dim = hidden / heads;
    EXPECT_EQ(cache.key.shape,
              (std::vector<std::size_t>{1, heads, 5, head_dim}));
    EXPECT_EQ(cache.value.shape,
              (std::vector<std::size_t>{1, heads, 5, head_dim}));
}

// A fixed capacity bounds the cache: filling it exactly succeeds, and the
// append that would exceed it throws.
TEST(AttentionTest, CapacityBoundsCache) {
    const AttentionWeights<float> weights = identity_weights(2, 1);
    KvCache<float> cache;
    cache.capacity = 2;
    GlobalAttention<float> attention(weights, cache);

    attention.forward(make_input(2, 2));
    EXPECT_THROW(attention.forward(make_input(1, 2)), std::length_error);
}

// Sliding window of 2 over 4 tokens: token 3 may see only tokens 2 and 3
// (the window includes the query's own position). Perturbing token 0 must
// leave row 3 unchanged; perturbing token 2 must change it. Pins the
// relative <= -window boundary without needing 256 real tokens.
TEST(AttentionTest, LocalWindowBoundary) {
    const std::size_t window = 2;
    const AttentionWeights<float> weights = identity_weights(2, 1);

    const auto run = [&](float token0_first, float token2_first) {
        std::vector<float> values = {token0_first, 0.1F, 0.2F,  -0.3F,
                                     token2_first, 0.5F, -0.6F, 0.7F};
        KvCache<float> cache;
        LocalAttention<float> attention(weights, cache, window);
        return attention.forward(make_tensor({1, 4, 2}, std::move(values)));
    };

    const Tensor<float> base = run(0.4F, -0.8F);
    const Tensor<float> moved_t0 = run(-1.5F, -0.8F);
    const Tensor<float> moved_t2 = run(0.4F, 1.1F);

    expect_near(row(moved_t0, 3), row(base, 3), kTight);
    EXPECT_TRUE(rows_differ(row(moved_t2, 3), row(base, 3)));
}

// Processing N tokens in one prefill call must match feeding them one at a
// time through the cache. This is the integration gate for query_start,
// append_kv_cache ordering, and context reading the cache rather than the
// new tokens. Failure signatures: every decode row wrong -> context uses
// qkv instead of the cache; rows shifted by one -> query_start off-by-one;
// only the local variant failing -> window offset vs query_start.
template <typename MakeAttention>
void expect_prefill_matches_decode(const AttentionWeights<float>& weights,
                                   MakeAttention make_attention) {
    const std::size_t tokens = 5;
    const std::size_t hidden = weights.output_bias.shape[0];
    const Tensor<float> input = make_input(tokens, hidden);

    KvCache<float> prefill_cache;
    auto prefill_attention = make_attention(weights, prefill_cache);
    const Tensor<float> prefill_out = prefill_attention.forward(input);

    KvCache<float> decode_cache;
    auto decode_attention = make_attention(weights, decode_cache);
    for (std::size_t i = 0; i < tokens; ++i) {
        const Tensor<float> token = make_tensor({1, 1, hidden}, row(input, i));
        const Tensor<float> decode_out = decode_attention.forward(token);
        expect_near(row(decode_out, 0), row(prefill_out, i), kTight);
    }
}

TEST(AttentionTest, PrefillMatchesDecodeGlobal) {
    expect_prefill_matches_decode(
        pattern_weights(4, 2),
        [](const AttentionWeights<float>& weights, KvCache<float>& cache) {
            return GlobalAttention<float>(weights, cache);
        });
}

// Window of 2 with 5 tokens, so the window clips mid-sequence and the
// local mask's cached-token offset is actually exercised.
TEST(AttentionTest, PrefillMatchesDecodeLocal) {
    expect_prefill_matches_decode(
        pattern_weights(4, 2),
        [](const AttentionWeights<float>& weights, KvCache<float>& cache) {
            return LocalAttention<float>(weights, cache, 2);
        });
}

} // namespace
} // namespace inference_engine
