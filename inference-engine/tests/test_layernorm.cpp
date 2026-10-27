// layernorm() unit tests: hand-computed rows, the optional shift, and the
// per-feature shape check.
#include "inference_engine/operator/layernorm.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

namespace inference_engine {

namespace {

constexpr float kTight = 1e-5F;

} // namespace

// Each row is normalized on its own, then scaled and shifted per feature.
TEST(LayerNormTest, NormalizesEachRowThenAppliesGainAndShift) {
    // row 0: mean 2.5, variance 1.25; row 1: constant, so eps keeps it at
    // zero (eps shifts row 0 by under 1e-5)
    const Tensor<float> input{{2, 4},
                              {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 5.0F, 5.0F, 5.0F}};
    const Tensor<float> weight{{4}, {1.0F, 2.0F, 1.0F, 1.0F}};
    const Tensor<float> bias{{4}, {0.0F, 0.0F, 0.0F, 1.0F}};

    const Tensor<float> output =
        layernorm<float>(input, weight, std::cref(bias));

    ASSERT_EQ(output.shape, input.shape);
    const float s = 1.0F / std::sqrt(1.25F);
    const std::vector<float> expected{
        -1.5F * s, -1.0F * s, 0.5F * s, (1.5F * s) + 1.0F,
        0.0F,      0.0F,      0.0F,     1.0F};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        EXPECT_NEAR(output.values[index], expected[index], kTight) << index;
    }
}

// Without a bias, layernorm() only normalizes and applies the gain.
TEST(LayerNormTest, ShiftIsOptional) {
    const Tensor<float> input{{3}, {-1.0F, 0.0F, 1.0F}};
    const Tensor<float> weight{{3}, {1.0F, 1.0F, 1.0F}};

    const Tensor<float> output = layernorm(input, weight);

    // variance 2/3 plus eps
    const float s = 1.0F / std::sqrt((2.0F / 3.0F) + 1e-5F);
    EXPECT_NEAR(output.values[0], -s, kTight);
    EXPECT_NEAR(output.values[1], 0.0F, kTight);
    EXPECT_NEAR(output.values[2], s, kTight);
}

// A gain whose width differs from the input's last dimension is rejected.
TEST(LayerNormTest, RejectsGainOfTheWrongWidth) {
    const Tensor<float> input{{2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};
    const Tensor<float> weight{{2}, {1.0F, 1.0F}};

    EXPECT_THROW(layernorm(input, weight), std::invalid_argument);
}

} // namespace inference_engine
