#include "inference_engine/operator/mlp.hpp"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

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

TEST(MLPTest, ForwardHandlesDifferentWidthsAndNonzeroBiases) {
    // input_size = 2, hidden_size = 3, two input rows.
    const inference_engine::Tensor<float> input{{2, 2},
                                                {1.0f, -1.0f, 0.5f, 2.0f}};
    // W_1 is [hidden_size, input_size] = [3, 2].
    const inference_engine::Tensor<float> w_1{
        {3, 2}, {1.0f, 2.0f, -1.0f, 0.5f, 0.0f, 1.0f}};
    const inference_engine::Tensor<float> b_1{{3}, {0.5f, -1.0f, 2.0f}};
    // W_2 is [input_size, hidden_size] = [2, 3].
    const inference_engine::Tensor<float> w_2{
        {2, 3}, {1.0f, 0.0f, -1.0f, 2.0f, 1.0f, 0.5f}};
    const inference_engine::Tensor<float> b_2{{2}, {-1.0f, 0.25f}};

    const inference_engine::MlpWeights weights{2, 3, w_1, b_1, w_2, b_2};
    const inference_engine::MLP mlp(weights);

    const auto output = mlp.forward(input);

    // Hidden pre-activations (W_1 x + b_1), worked by hand:
    //   row 0: x = [1, -1]  -> [-0.5, -2.5, 1]
    //   row 1: x = [0.5, 2] -> [5, -0.5, 4]
    const float h00 = gelu_tanh(-0.5f);
    const float h01 = gelu_tanh(-2.5f);
    const float h02 = gelu_tanh(1.0f);
    const float h10 = gelu_tanh(5.0f);
    const float h11 = gelu_tanh(-0.5f);
    const float h12 = gelu_tanh(4.0f);

    ASSERT_EQ(output.shape, std::vector<std::size_t>({2, 2}));
    EXPECT_NEAR(output.values[0], h00 - h02 - 1.0f, 1e-6f);
    EXPECT_NEAR(output.values[1], 2.0f * h00 + h01 + 0.5f * h02 + 0.25f, 1e-6f);
    EXPECT_NEAR(output.values[2], h10 - h12 - 1.0f, 1e-6f);
    EXPECT_NEAR(output.values[3], 2.0f * h10 + h11 + 0.5f * h12 + 0.25f, 1e-6f);
}
