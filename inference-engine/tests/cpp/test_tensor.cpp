#include "inference_engine/operator/activation.hpp"
#include "inference_engine/operator/attention.hpp"
#include "inference_engine/tensor.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace inference_engine {

static_assert(std::is_base_of_v<Attention<float>, GlobalAttention<float>>);
static_assert(
    std::is_base_of_v<Attention<std::int8_t>, LocalAttention<std::int8_t>>);
static_assert(
    std::is_same_v<decltype(softmax(std::declval<const Tensor<std::int8_t>&>(),
                                    std::size_t{})),
                   Tensor<std::int8_t>>);

TEST(MatmulTest, MultipliesCompatibleMatrices) {
    const Tensor<float> left{{2, 3}, {1, 2, 3, 4, 5, 6}};
    const Tensor<float> right{{3, 2}, {7, 8, 9, 10, 11, 12}};

    const auto result = matmul(left, right);

    const std::vector<std::size_t> expected_shape{2, 2};
    const std::vector<float> expected_values{58, 64, 139, 154};
    EXPECT_EQ(result.shape, expected_shape);
    EXPECT_EQ(result.values, expected_values);
}

TEST(MatmulTest, AccumulatesInt8ValuesBeforeNarrowing) {
    const Tensor<std::int8_t> left{{1, 2}, {100, 100}};
    const Tensor<std::int8_t> right{{2, 1}, {2, -2}};

    const auto result = matmul(left, right);

    EXPECT_EQ(result.shape, (std::vector<std::size_t>{1, 1}));
    EXPECT_EQ(result.values, (std::vector<std::int8_t>{0}));
}

TEST(MatmulTest, RejectsInt8ResultOverflow) {
    const Tensor<std::int8_t> left{{1, 1}, {127}};
    const Tensor<std::int8_t> right{{1, 1}, {2}};

    EXPECT_THROW(matmul(left, right), std::overflow_error);
}

TEST(MatmulTest, RejectsIncompatibleDimensions) {
    const Tensor<float> left{{2, 3}, {1, 2, 3, 4, 5, 6}};
    const Tensor<float> right{{2, 2}, {1, 2, 3, 4}};

    EXPECT_THROW(matmul(left, right), std::invalid_argument);
}

TEST(MatmulTest, RejectsInvalidStorage) {
    const Tensor<float> left{{2, 3}, {1, 2}};
    const Tensor<float> right{{3, 2}, {7, 8, 9, 10, 11, 12}};

    EXPECT_THROW(matmul(left, right), std::invalid_argument);
}

TEST(TensorTest, StoresOtherScalarTypes) {
    const Tensor<std::uint8_t> ids{{2}, {1, 2}};

    EXPECT_EQ(ids.shape, (std::vector<std::size_t>{2}));
    EXPECT_EQ(ids.values, (std::vector<std::uint8_t>{1, 2}));
}

// Validation counts values across any number of dimensions.
TEST(TensorTest, ValidatesValueCountForAnyRank) {
    EXPECT_TRUE(valid_tensor(Tensor<float>{{2, 3, 2}, std::vector<float>(12)}));
    EXPECT_FALSE(
        valid_tensor(Tensor<float>{{2, 3, 2}, std::vector<float>(11)}));
    EXPECT_TRUE(valid_tensor(Tensor<float>{{4}, {1, 2, 3, 4}}));
    EXPECT_FALSE(valid_tensor(Tensor<float>{{4}, {1, 2, 3}}));
}

// An empty shape is a scalar with exactly one value.
TEST(TensorTest, TreatsEmptyShapeAsScalar) {
    EXPECT_TRUE(valid_tensor(Tensor<float>{{}, {1}}));
    EXPECT_FALSE(valid_tensor(Tensor<float>{{}, {}}));
}

// A zero dimension is rejected, even beside huge dimensions.
TEST(TensorTest, RejectsZeroDimension) {
    constexpr auto kHuge = std::numeric_limits<std::size_t>::max();

    EXPECT_FALSE(valid_tensor(Tensor<float>{{0, 3}, {}}));
    EXPECT_FALSE(valid_tensor(Tensor<float>{{0, 3}, {1}}));
    EXPECT_FALSE(valid_tensor(Tensor<float>{{kHuge, kHuge, 0}, {}}));
}

// A shape whose element count overflows is rejected rather than wrapping.
TEST(TensorTest, RejectsShapeProductOverflow) {
    constexpr auto kMax = std::numeric_limits<std::size_t>::max();
    // 3 * kRows wraps to exactly 2, so an unchecked product would match.
    constexpr auto kRows = (kMax / 3) + 1;

    EXPECT_FALSE(valid_tensor(Tensor<float>{{kRows, 3}, {1, 2}}));
    EXPECT_FALSE(valid_matrix(Tensor<float>{{kRows, 3}, {1, 2}}));
    EXPECT_FALSE(valid_tensor(Tensor<float>{{kMax, kMax}, {}}));
}

// Matrices must be two-dimensional with nonzero dimensions.
TEST(TensorTest, MatrixRequiresTwoNonzeroDimensions) {
    EXPECT_TRUE(valid_matrix(Tensor<float>{{2, 2}, {1, 2, 3, 4}}));
    EXPECT_FALSE(valid_matrix(Tensor<float>{{4}, {1, 2, 3, 4}}));
    EXPECT_FALSE(valid_matrix(Tensor<float>{{1, 2, 2}, {1, 2, 3, 4}}));
    EXPECT_FALSE(valid_matrix(Tensor<float>{{0, 2}, {}}));
}

} // namespace inference_engine
