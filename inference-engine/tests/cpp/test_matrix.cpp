#include "inference_engine/operator/matrix.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace inference_engine {

TEST(MatrixTest, MultipliesCompatibleMatrices) {
    const Tensor<float> left{{2, 3}, {1, 2, 3, 4, 5, 6}};
    const Tensor<float> right{{3, 2}, {7, 8, 9, 10, 11, 12}};

    const auto result = matmul(left, right);

    const std::vector<std::size_t> expected_shape{2, 2};
    const std::vector<float> expected_values{58, 64, 139, 154};
    EXPECT_EQ(result.shape, expected_shape);
    EXPECT_EQ(result.values, expected_values);
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
