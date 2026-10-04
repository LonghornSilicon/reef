#include "inference_engine/operator/mlp.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

float gelu_tanh(float x) {
    constexpr float kCoefficient = 0.7978845608028654f;
    constexpr float kCubic = 0.044715f;
    return 0.5f * x *
           (1.0f + std::tanh(kCoefficient * (x + kCubic * x * x * x)));
}

} // namespace

TEST(MLPTest, ForwardAppliesTwoLinearLayersWithGelu) {
    const inference_engine::Tensor<float> input{{2, 2},
                                                {1.0f, 2.0f, 3.0f, 4.0f}};
    const inference_engine::Tensor<float> w_1{{2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}};
    const inference_engine::Tensor<float> b_1{{2}, {0.0f, 0.0f}};
    const inference_engine::Tensor<float> w_2{{2, 2}, {1.0f, 1.0f, 0.0f, 1.0f}};
    const inference_engine::Tensor<float> b_2{{2}, {0.0f, 0.0f}};

    const inference_engine::MlpWeights weights{2, 2, w_1, b_1, w_2, b_2};
    const inference_engine::MLP mlp(weights);

    const auto output = mlp.forward(input);

    ASSERT_EQ(output.shape, std::vector<std::size_t>({2, 2}));
    EXPECT_NEAR(output.values[0], gelu_tanh(1.0f) + gelu_tanh(2.0f), 1e-6f);
    EXPECT_NEAR(output.values[1], gelu_tanh(2.0f), 1e-6f);
    EXPECT_NEAR(output.values[2], gelu_tanh(3.0f) + gelu_tanh(4.0f), 1e-6f);
    EXPECT_NEAR(output.values[3], gelu_tanh(4.0f), 1e-6f);
}

namespace inference_engine {

TEST(MlpTest, MatchesFixedReferenceOutput) {
    // GPT-Neo permits an explicit intermediate width.
    // Here hidden_size = 2 and intermediate_size = 3.
    const MlpWeights weights{
        2,
        3,
        Tensor<float>{{3, 2}, {1, 1, 2, 1, 0, -1}},
        Tensor<float>{{3}, {0, 0, 0}},
        Tensor<float>{{2, 3}, {1, 2, 3, -2, 1, 0.5F}},
        Tensor<float>{{2}, {0.25F, -0.5F}},
    };
    const MLP mlp(weights);
    const Tensor<float> input{{1, 2}, {1, -2}};

    const auto output = mlp.forward(input);

    // First projection: [-1, 0, 2].
    // Independently evaluated using the reference tanh-GELU formula.
    ASSERT_EQ(output.shape, (std::vector<std::size_t>{1, 2}));
    ASSERT_EQ(output.values.size(), 2U);
    EXPECT_NEAR(output.values[0], 5.954985073F, 1e-5F);
    EXPECT_NEAR(output.values[1], 0.794914866F, 1e-5F);
}

TEST(MlpTest, AppliesBothBiasesAcrossMultipleRows) {
    // Default GPT-Neo expansion: hidden_size = 2,
    // intermediate_size = 4 * hidden_size = 8.
    const MlpWeights weights{
        2,
        8,
        Tensor<float>{{8, 2}, std::vector<float>(16, 0.0F)},
        Tensor<float>{{8}, {1, 0, 0, 0, 0, 0, 0, 0}},
        Tensor<float>{{2, 8},
                      {2, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0}},
        Tensor<float>{{2}, {0.5F, -0.25F}},
    };
    const MLP mlp(weights);
    const Tensor<float> input{{3, 2}, {0, 0, 1, -2, -3, 4}};

    const auto output = mlp.forward(input);

    // Zero first-layer weights make every row identical.
    // GELU_tanh(1) = 0.8411919906082768.
    // Output = [2 * GELU_tanh(1) + 0.5,
    //           -GELU_tanh(1) - 0.25].
    ASSERT_EQ(output.shape, (std::vector<std::size_t>{3, 2}));
    ASSERT_EQ(output.values.size(), 6U);
    for (std::size_t row = 0; row < 3; ++row) {
        EXPECT_NEAR(output.values[row * 2], 2.182383981F, 1e-5F);
        EXPECT_NEAR(output.values[row * 2 + 1], -1.091191991F, 1e-5F);
    }
}

TEST(MlpTest, SupportsTinyStoriesInstruct28MDimensions) {
    // Matches GPT_NEO_CONFIGS["TinyStories-Instruct-28M"].
    constexpr std::size_t kHidden = 512;
    constexpr std::size_t kIntermediate = 4 * kHidden;

    const MlpWeights weights{
        kHidden,
        kIntermediate,
        Tensor<float>{{kIntermediate, kHidden},
                      std::vector<float>(kIntermediate * kHidden, 0.0F)},
        Tensor<float>{{kIntermediate}, std::vector<float>(kIntermediate, 0.0F)},
        Tensor<float>{{kHidden, kIntermediate},
                      std::vector<float>(kHidden * kIntermediate, 0.0F)},
        Tensor<float>{{kHidden}, std::vector<float>(kHidden, 0.25F)},
    };
    const MLP mlp(weights);
    const Tensor<float> input{{1, kHidden}, std::vector<float>(kHidden, 1.0F)};

    const auto output = mlp.forward(input);

    // Zero projections leave only the second-layer bias.
    ASSERT_EQ(output.shape, (std::vector<std::size_t>{1, kHidden}));
    ASSERT_EQ(output.values.size(), kHidden);
    for (const float value : output.values) {
        EXPECT_FLOAT_EQ(value, 0.25F);
    }
}

namespace {
MlpWeights make_test_weights() {
    return {2,
            3,
            Tensor<float>{{3, 2}, {1, 1, 2, 1, 0, -1}},
            Tensor<float>{{3}, {0.5F, -0.25F, 1.0F}},
            Tensor<float>{{2, 3}, {1, 2, 3, -2, 1, 0.5F}},
            Tensor<float>{{2}, {0.25F, -0.5F}}};
}

constexpr Tensor<float> MlpWeights::* kParameters[] = {
    &MlpWeights::W_1, &MlpWeights::b_1, &MlpWeights::W_2, &MlpWeights::b_2};
} // namespace

TEST(MlpTest, RejectsWrongInputRank) {
    const MlpWeights weights = make_test_weights();
    const MLP mlp(weights);
    const Tensor<float> input{{2}, {1, -2}};
    EXPECT_THROW(static_cast<void>(mlp.forward(input)), std::invalid_argument);
}

TEST(MlpTest, RejectsWrongInputWidth) {
    const MlpWeights weights = make_test_weights();
    const MLP mlp(weights);
    const Tensor<float> input{{1, 3}, {1, -2, 3}};
    EXPECT_THROW(static_cast<void>(mlp.forward(input)), std::invalid_argument);
}

TEST(MlpTest, RejectsShortInputStorage) {
    const MlpWeights weights = make_test_weights();
    const MLP mlp(weights);
    const Tensor<float> input{{2, 2}, {1, -2, 3}};
    EXPECT_THROW(static_cast<void>(mlp.forward(input)), std::invalid_argument);
}

TEST(MlpTest, RejectsExcessInputStorage) {
    const MlpWeights weights = make_test_weights();
    const MLP mlp(weights);
    const Tensor<float> input{{1, 2}, {1, -2, 3}};
    EXPECT_THROW(static_cast<void>(mlp.forward(input)), std::invalid_argument);
}

TEST(MlpTest, RejectsMalformedParameterShapes) {
    for (const auto parameter : kParameters) {
        MlpWeights weights = make_test_weights();
        // Preserve the values so rejection must come from the shape.
        (weights.*parameter).shape = {1, 1, 1};
        const MLP mlp(weights);
        EXPECT_THROW(
            static_cast<void>(mlp.forward(Tensor<float>{{1, 2}, {1, -2}})),
            std::invalid_argument);
    }
}

TEST(MlpTest, RejectsShortParameterStorage) {
    for (const auto parameter : kParameters) {
        MlpWeights weights = make_test_weights();
        // Every fixture parameter has at least two values, so the bias
        // remains present after removing one value.
        (weights.*parameter).values.pop_back();
        const MLP mlp(weights);
        EXPECT_THROW(
            static_cast<void>(mlp.forward(Tensor<float>{{1, 2}, {1, -2}})),
            std::invalid_argument);
    }
}

TEST(MlpTest, RepeatedCallsPreserveInputsAndWeights) {
    const MlpWeights weights = make_test_weights();
    const MlpWeights original_weights = weights;
    const MLP mlp(weights);
    const Tensor<float> input{{1, 2}, {1, -2}};
    const Tensor<float> original_input = input;
    const Tensor<float> other_input{{2, 2}, {3, 4, -1, 2}};
    const Tensor<float> original_other_input = other_input;

    const auto first = mlp.forward(input);
    const auto other_output = mlp.forward(other_input);
    const auto repeated = mlp.forward(input);

    ASSERT_EQ(other_output.shape, (std::vector<std::size_t>{2, 2}));
    ASSERT_EQ(other_output.values.size(), 4U);
    EXPECT_EQ(repeated.shape, first.shape);
    EXPECT_EQ(repeated.values, first.values);
    EXPECT_EQ(input.shape, original_input.shape);
    EXPECT_EQ(input.values, original_input.values);
    EXPECT_EQ(other_input.shape, original_other_input.shape);
    EXPECT_EQ(other_input.values, original_other_input.values);
    EXPECT_EQ(weights.input_size, original_weights.input_size);
    EXPECT_EQ(weights.hidden_size, original_weights.hidden_size);
    for (const auto parameter : kParameters) {
        EXPECT_EQ((weights.*parameter).shape,
                  (original_weights.*parameter).shape);
        EXPECT_EQ((weights.*parameter).values,
                  (original_weights.*parameter).values);
    }
}

} // namespace inference_engine
