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
    const inference_engine::Tensor<float> fc_weight{{2, 2},
                                                    {1.0f, 0.0f, 0.0f, 1.0f}};
    const inference_engine::Tensor<float> fc_bias{{2}, {0.0f, 0.0f}};
    const inference_engine::Tensor<float> proj_weight{{2, 2},
                                                      {1.0f, 1.0f, 0.0f, 1.0f}};
    const inference_engine::Tensor<float> proj_bias{{2}, {0.0f, 0.0f}};

    const inference_engine::MlpWeights weights{2,       2,           fc_weight,
                                               fc_bias, proj_weight, proj_bias};
    const inference_engine::MLP mlp(weights);

    const auto output = mlp.forward(input);

    ASSERT_EQ(output.shape, std::vector<std::size_t>({2, 2}));
    EXPECT_NEAR(output.values[0], gelu_tanh(1.0f) + gelu_tanh(2.0f), 1e-6f);
    EXPECT_NEAR(output.values[1], gelu_tanh(2.0f), 1e-6f);
    EXPECT_NEAR(output.values[2], gelu_tanh(3.0f) + gelu_tanh(4.0f), 1e-6f);
    EXPECT_NEAR(output.values[3], gelu_tanh(4.0f), 1e-6f);
}
