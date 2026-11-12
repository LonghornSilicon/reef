// TpGptNeo against the single-chip engine on the reference tests' tiny random
// model: column-only is bit-exact for any rank count, Megatron is bit-exact
// with one rank and close with more, and each scheme syncs where it should.
#include "dist_inference/tp_model.hpp"

#include "inference_engine/model.hpp"
#include "inference_engine/weights.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace dist_inference {

namespace {

using inference_engine::GPTNeo;
using inference_engine::GptNeoConfig;

// vocab 64, 32 positions, width 32, 2 layers (global then local), 4 heads,
// MLP width 128, window 4: test_gpt_neo.cpp's model.
constexpr GptNeoConfig kTiny{64, 32, 32, 2, 4, 128, 4, 1e-5F};
constexpr std::uint32_t kSeed = 7;

// Longer than the window, so the local layer drops positions.
const TokenIds prompt_tokens{3, 17, 42, 5, 63, 0, 9, 9, 28, 51};

ModelWeights<float> tiny_weights(bool w8a8) {
    ModelWeights<float> weights =
        inference_engine::random_gpt_neo(kTiny, kSeed);
    if (w8a8) {
        inference_engine::quantize_w8a8(weights);
    }
    return weights;
}

/// Logits after each token of the prompt, fed one at a time.
template <typename Model> std::vector<Tensor<float>> decode(Model& model) {
    std::vector<Tensor<float>> logits;
    for (const std::uint32_t id : prompt_tokens) {
        logits.push_back(model.forward({id}));
    }
    return logits;
}

/// Largest |a - b| over the largest |a|.
float relative_error(const Tensor<float>& exact, const Tensor<float>& other) {
    float largest = 0.0F;
    float worst = 0.0F;
    for (std::size_t i = 0; i < exact.values.size(); ++i) {
        largest = std::max(largest, std::fabs(exact.values[i]));
        worst = std::max(worst, std::fabs(exact.values[i] - other.values[i]));
    }
    return worst / largest;
}

/// True if two sync lists match field by field.
bool same_syncs(const std::vector<Sync>& actual,
                const std::vector<Sync>& expected) {
    return std::equal(actual.begin(), actual.end(), expected.begin(),
                      expected.end(), [](const Sync& a, const Sync& b) {
                          return a.kind == b.kind && a.rows == b.rows &&
                                 a.width == b.width;
                      });
}

} // namespace

// Column-only gives the single chip's logits bit for bit for 1, 2 and 4
// ranks, in float and W8A8, for a prefill and for every decode step.
TEST(TpGptNeoTest, ColumnOnlyIsBitExactForAnyRanks) {
    for (const bool w8a8 : {false, true}) {
        const ModelWeights<float> weights = tiny_weights(w8a8);
        GPTNeo<float> single(weights);
        const Tensor<float> prefill = single.forward(prompt_tokens);
        single.reset();
        const std::vector<Tensor<float>> steps = decode(single);
        for (const std::size_t ranks : {1U, 2U, 4U}) {
            TpGptNeo<float> split(weights, ranks, TpScheme::kColumnOnly);
            EXPECT_EQ(split.forward(prompt_tokens).values, prefill.values)
                << w8a8 << ' ' << ranks;
            split.reset();
            const std::vector<Tensor<float>> split_steps = decode(split);
            for (std::size_t step = 0; step < steps.size(); ++step) {
                EXPECT_EQ(split_steps[step].values, steps[step].values)
                    << w8a8 << ' ' << ranks << ' ' << step;
            }
        }
    }
}

// Megatron with one rank is the single chip bit for bit.
TEST(TpGptNeoTest, MegatronWithOneRankIsTheSingleChip) {
    for (const bool w8a8 : {false, true}) {
        const ModelWeights<float> weights = tiny_weights(w8a8);
        GPTNeo<float> single(weights);
        TpGptNeo<float> split(weights, 1, TpScheme::kMegatron);
        EXPECT_EQ(split.forward(prompt_tokens).values,
                  single.forward(prompt_tokens).values)
            << w8a8;
    }
}

// Megatron with 2 and 4 ranks changes the logits by rounding only: within
// 1e-5 of the largest logit in float and 5% in W8A8, where each rank also
// quantizes its own slice. It must differ somewhere, or the split never ran.
TEST(TpGptNeoTest, MegatronStaysCloseToTheSingleChip) {
    for (const bool w8a8 : {false, true}) {
        const ModelWeights<float> weights = tiny_weights(w8a8);
        GPTNeo<float> single(weights);
        const Tensor<float> exact = single.forward(prompt_tokens);
        for (const std::size_t ranks : {2U, 4U}) {
            TpGptNeo<float> split(weights, ranks, TpScheme::kMegatron);
            const float error =
                relative_error(exact, split.forward(prompt_tokens));
            EXPECT_GT(error, 0.0F) << w8a8 << ' ' << ranks;
            EXPECT_LT(error, w8a8 ? 0.05F : 1e-5F) << w8a8 << ' ' << ranks;
        }
    }
}

// Megatron's prefill and decode stay bit-identical: the split changes the
// sums, not their dependence on other rows.
TEST(TpGptNeoTest, MegatronPrefillMatchesDecodeBitwise) {
    const ModelWeights<float> weights = tiny_weights(false);
    TpGptNeo<float> decoded(weights, 2);
    const std::vector<Tensor<float>> steps = decode(decoded);
    TpGptNeo<float> prefilled(weights, 2);
    EXPECT_EQ(prefilled.forward(prompt_tokens).values, steps.back().values);
}

// Column-only syncs four times per layer (attention heads, attention output,
// c_fc, c_proj) and gathers the logits last.
TEST(TpGptNeoTest, ColumnOnlySyncsFourTimesPerLayer) {
    const ModelWeights<float> weights = tiny_weights(false);
    const std::size_t rows = prompt_tokens.size();
    const std::size_t hidden = kTiny.hidden_size;
    std::vector<Sync> expected;
    for (std::size_t layer = 0; layer < kTiny.num_layers; ++layer) {
        for (const std::size_t width :
             {hidden, hidden, kTiny.inner_size, hidden}) {
            expected.push_back({Sync::Kind::kGather, rows, width});
        }
    }
    expected.push_back({Sync::Kind::kGather, 1, kTiny.vocab_size});

    TpGptNeo<float> model(weights, 2, TpScheme::kColumnOnly);
    static_cast<void>(model.forward(prompt_tokens));
    EXPECT_TRUE(same_syncs(model.syncs(), expected));
}

// Megatron syncs twice per layer, summing the attention output and c_proj,
// and gathers the logits last.
TEST(TpGptNeoTest, MegatronSyncsTwicePerLayer) {
    const ModelWeights<float> weights = tiny_weights(false);
    const std::size_t rows = prompt_tokens.size();
    std::vector<Sync> expected;
    for (std::size_t layer = 0; layer < kTiny.num_layers; ++layer) {
        expected.push_back({Sync::Kind::kSum, rows, kTiny.hidden_size});
        expected.push_back({Sync::Kind::kSum, rows, kTiny.hidden_size});
    }
    expected.push_back({Sync::Kind::kGather, 1, kTiny.vocab_size});

    TpGptNeo<float> model(weights, 2, TpScheme::kMegatron);
    static_cast<void>(model.forward(prompt_tokens));
    EXPECT_TRUE(same_syncs(model.syncs(), expected));
}

// Column-only generates the single chip's tokens, and repeats across calls.
TEST(TpGptNeoTest, ColumnOnlyGeneratesTheSingleChipTokens) {
    const ModelWeights<float> weights = tiny_weights(false);
    GPTNeo<float> single(weights);
    TpGptNeo<float> split(weights, 4, TpScheme::kColumnOnly);

    const TokenIds expected = single.generate(prompt_tokens, 8);
    EXPECT_EQ(split.generate(prompt_tokens, 8), expected);
    EXPECT_EQ(split.generate(prompt_tokens, 8), expected);
    EXPECT_TRUE(split.generate(prompt_tokens, 0).empty());
}

// Zero ranks, and ranks that do not divide the 4 heads, are rejected.
TEST(TpGptNeoTest, RejectsRanksThatDoNotSplitTheHeads) {
    const ModelWeights<float> weights = tiny_weights(false);
    EXPECT_THROW(TpGptNeo<float>(weights, 0), std::invalid_argument);
    EXPECT_THROW(TpGptNeo<float>(weights, 3), std::invalid_argument);
    EXPECT_THROW(TpGptNeo<float>(weights, 8), std::invalid_argument);
}

} // namespace dist_inference
