// MLP unit tests, ported from the original MLP change onto the templated
// MlpWeights/MLP API. The second case uses a different inner width and
// nonzero biases on both projections, which also pins linear()'s weight
// layout: the [inner_size, input_size] c_fc weight is non-square.
#include "inference_engine/operator/mlp.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace inference_engine {

namespace {

constexpr float kTight = 1e-6F;

/// Independent GELU reference, so the tests do not reuse the implementation.
float gelu_tanh(float x) {
    constexpr float kCoefficient = 0.7978845608028654F;
    constexpr float kCubic = 0.044715F;
    return 0.5F * x *
           (1.0F + std::tanh(kCoefficient * (x + (kCubic * x * x * x))));
}

} // namespace

TEST(MLPTest, ForwardAppliesTwoProjectionsWithGelu) {
    const Tensor<float> input{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
    const MlpWeights<float> weights{
        2,
        2,
        Tensor<float>{{2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}},
        Tensor<float>{{2}, {0.0F, 0.0F}},
        Tensor<float>{{2, 2}, {1.0F, 1.0F, 0.0F, 1.0F}},
        Tensor<float>{{2}, {0.0F, 0.0F}},
    };
    const MLP<float> mlp(weights);

    const Tensor<float> output = mlp.forward(input);

    ASSERT_EQ(output.shape, std::vector<std::size_t>({2, 2}));
    EXPECT_NEAR(output.values[0], gelu_tanh(1.0F) + gelu_tanh(2.0F), kTight);
    EXPECT_NEAR(output.values[1], gelu_tanh(2.0F), kTight);
    EXPECT_NEAR(output.values[2], gelu_tanh(3.0F) + gelu_tanh(4.0F), kTight);
    EXPECT_NEAR(output.values[3], gelu_tanh(4.0F), kTight);
}

TEST(MLPTest, ForwardHandlesDifferentWidthsAndNonzeroBiases) {
    // input_size = 2, inner_size = 3, two input rows.
    const MlpWeights<float> weights{
        2,
        3,
        // c_fc_weight is [inner_size, input_size] = [3, 2].
        Tensor<float>{{3, 2}, {1.0F, 2.0F, -1.0F, 0.5F, 0.0F, 1.0F}},
        Tensor<float>{{3}, {0.5F, -1.0F, 2.0F}},
        // c_proj_weight is [input_size, inner_size] = [2, 3].
        Tensor<float>{{2, 3}, {1.0F, 0.0F, -1.0F, 2.0F, 1.0F, 0.5F}},
        Tensor<float>{{2}, {-1.0F, 0.25F}},
    };
    const MLP<float> mlp(weights);

    const Tensor<float> output =
        mlp.forward(Tensor<float>{{2, 2}, {1.0F, -1.0F, 0.5F, 2.0F}});

    // Inner pre-activations (c_fc x + bias), worked by hand:
    //   row 0: x = [1, -1]  -> [-0.5, -2.5, 1]
    //   row 1: x = [0.5, 2] -> [5, -0.5, 4]
    const float h00 = gelu_tanh(-0.5F);
    const float h01 = gelu_tanh(-2.5F);
    const float h02 = gelu_tanh(1.0F);
    const float h10 = gelu_tanh(5.0F);
    const float h11 = gelu_tanh(-0.5F);
    const float h12 = gelu_tanh(4.0F);

    ASSERT_EQ(output.shape, std::vector<std::size_t>({2, 2}));
    EXPECT_NEAR(output.values[0], h00 - h02 - 1.0F, kTight);
    EXPECT_NEAR(output.values[1], (2.0F * h00) + h01 + (0.5F * h02) + 0.25F,
                kTight);
    EXPECT_NEAR(output.values[2], h10 - h12 - 1.0F, kTight);
    EXPECT_NEAR(output.values[3], (2.0F * h10) + h11 + (0.5F * h12) + 0.25F,
                kTight);
}

TEST(MLPTest, OmitsBiasWhenTheBiasTensorIsEmpty) {
    const MlpWeights<float> weights{
        2,
        2,
        Tensor<float>{{2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}},
        Tensor<float>{},
        Tensor<float>{{2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}},
        Tensor<float>{},
    };
    const MLP<float> mlp(weights);

    const Tensor<float> output =
        mlp.forward(Tensor<float>{{1, 2}, {1.0F, 2.0F}});

    ASSERT_EQ(output.shape, std::vector<std::size_t>({1, 2}));
    EXPECT_NEAR(output.values[0], gelu_tanh(1.0F), kTight);
    EXPECT_NEAR(output.values[1], gelu_tanh(2.0F), kTight);
}

TEST(MLPTest, RejectsInputThatIsNotTwoDimensional) {
    const MlpWeights<float> weights{
        2,
        2,
        Tensor<float>{{2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}},
        Tensor<float>{},
        Tensor<float>{{2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}},
        Tensor<float>{},
    };
    const MLP<float> mlp(weights);

    EXPECT_THROW(
        static_cast<void>(mlp.forward(Tensor<float>{{2}, {1.0F, 2.0F}})),
        std::invalid_argument);
}

TEST(MLPTest, RejectsInputWidthMismatch) {
    const MlpWeights<float> weights{
        2,
        2,
        Tensor<float>{{2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}},
        Tensor<float>{},
        Tensor<float>{{2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}},
        Tensor<float>{},
    };
    const MLP<float> mlp(weights);

    EXPECT_THROW(static_cast<void>(
                     mlp.forward(Tensor<float>{{1, 3}, {1.0F, 2.0F, 3.0F}})),
                 std::invalid_argument);
}

TEST(MLPTest, RejectsWeightsThatDoNotMatchTheConfiguredWidths) {
    const MlpWeights<float> weights{
        2,
        3,
        // Declares inner_size 3 but supplies a [2, 2] c_fc weight.
        Tensor<float>{{2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}},
        Tensor<float>{},
        Tensor<float>{{2, 3}, {1.0F, 0.0F, -1.0F, 2.0F, 1.0F, 0.5F}},
        Tensor<float>{},
    };
    const MLP<float> mlp(weights);

    EXPECT_THROW(
        static_cast<void>(mlp.forward(Tensor<float>{{1, 2}, {1.0F, 2.0F}})),
        std::invalid_argument);
}

} // namespace inference_engine
