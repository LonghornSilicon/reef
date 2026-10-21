// linear() unit tests. The weights here are deliberately non-square so the
// [out_features, in_features] layout is pinned: a transposed weight changes
// the inner dimension and is rejected, rather than silently projecting
// through the wrong axis the way a square weight would allow.
#include "inference_engine/operator/linear.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

namespace inference_engine {

namespace {

constexpr float kTight = 1e-6F;

/// Maps a width-2 input to width 3, so the layout is observable.
Tensor<float> wide_weight() {
    return Tensor<float>{{3, 2}, {1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F}};
}

} // namespace

TEST(LinearTest, ProjectsThroughTransposedWeight) {
    const Tensor<float> input{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};

    const Tensor<float> output = linear(input, wide_weight());

    ASSERT_EQ(output.shape, std::vector<std::size_t>({2, 3}));
    const std::vector<float> expected{1.0F, 2.0F, 3.0F, 3.0F, 4.0F, 7.0F};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        EXPECT_NEAR(output.values[index], expected[index], kTight) << index;
    }
}

TEST(LinearTest, AddsBiasPerOutputFeature) {
    const Tensor<float> input{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
    const Tensor<float> bias{{3}, {0.5F, -1.0F, 2.0F}};

    const Tensor<float> output =
        linear<float>(input, wide_weight(), std::cref(bias));

    ASSERT_EQ(output.shape, std::vector<std::size_t>({2, 3}));
    const std::vector<float> expected{1.5F, 1.0F, 5.0F, 3.5F, 3.0F, 9.0F};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        EXPECT_NEAR(output.values[index], expected[index], kTight) << index;
    }
}

TEST(LinearTest, RejectsWeightInTheOppositeLayout) {
    const Tensor<float> input{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
    // [in_features, out_features] instead of [out_features, in_features].
    const Tensor<float> transposed{{2, 3},
                                   {1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 1.0F}};

    EXPECT_THROW(linear(input, transposed), std::invalid_argument);
}

TEST(LinearTest, RejectsBiasOfTheWrongWidth) {
    const Tensor<float> input{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
    const Tensor<float> bias{{2}, {0.5F, -1.0F}};

    EXPECT_THROW(linear<float>(input, wide_weight(), std::cref(bias)),
                 std::invalid_argument);
}

TEST(LinearTest, RejectsTwoDimensionalBias) {
    const Tensor<float> input{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
    const Tensor<float> bias{{1, 3}, {0.5F, -1.0F, 2.0F}};

    EXPECT_THROW(linear<float>(input, wide_weight(), std::cref(bias)),
                 std::invalid_argument);
}

TEST(LinearTest, RejectsNonMatrixWeight) {
    const Tensor<float> input{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
    const Tensor<float> vector_weight{{2}, {1.0F, 1.0F}};

    EXPECT_THROW(linear(input, vector_weight), std::invalid_argument);
}

TEST(LinearTest, RejectsInputWidthMismatch) {
    const Tensor<float> input{{2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};

    EXPECT_THROW(linear(input, wide_weight()), std::invalid_argument);
}

} // namespace inference_engine
