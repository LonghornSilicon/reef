#include "inference_engine/operator/activation.hpp"
#include "inference_engine/operator/attention.hpp"
#include "inference_engine/operator/tensor.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
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

TEST(MatrixTest, MultipliesCompatibleMatrices) {
    const Tensor<float> left{{2, 3}, {1, 2, 3, 4, 5, 6}};
    const Tensor<float> right{{3, 2}, {7, 8, 9, 10, 11, 12}};

    const auto result = matmul(left, right);

    const std::vector<std::size_t> expected_shape{2, 2};
    const std::vector<float> expected_values{58, 64, 139, 154};
    EXPECT_EQ(result.shape, expected_shape);
    EXPECT_EQ(result.values, expected_values);
}

TEST(MatrixTest, AccumulatesInt8ValuesBeforeNarrowing) {
    const Tensor<std::int8_t> left{{1, 2}, {100, 100}};
    const Tensor<std::int8_t> right{{2, 1}, {2, -2}};

    const auto result = matmul(left, right);

    EXPECT_EQ(result.shape, (std::vector<std::size_t>{1, 1}));
    EXPECT_EQ(result.values, (std::vector<std::int8_t>{0}));
}

TEST(MatrixTest, RejectsInt8ResultOverflow) {
    const Tensor<std::int8_t> left{{1, 1}, {127}};
    const Tensor<std::int8_t> right{{1, 1}, {2}};

    EXPECT_THROW(matmul(left, right), std::overflow_error);
}

TEST(MatrixTest, RejectsIncompatibleDimensions) {
    const Tensor<float> left{{2, 3}, {1, 2, 3, 4, 5, 6}};
    const Tensor<float> right{{2, 2}, {1, 2, 3, 4}};

    EXPECT_THROW(matmul(left, right), std::invalid_argument);
}

TEST(MatrixTest, RejectsInvalidStorage) {
    const Tensor<float> left{{2, 3}, {1, 2}};
    const Tensor<float> right{{3, 2}, {7, 8, 9, 10, 11, 12}};

    EXPECT_THROW(matmul(left, right), std::invalid_argument);
}

TEST(TensorTest, StoresOtherScalarTypes) {
    const Tensor<std::uint8_t> ids{{2}, {1, 2}};

    EXPECT_EQ(ids.shape, (std::vector<std::size_t>{2}));
    EXPECT_EQ(ids.values, (std::vector<std::uint8_t>{1, 2}));
}

} // namespace inference_engine
