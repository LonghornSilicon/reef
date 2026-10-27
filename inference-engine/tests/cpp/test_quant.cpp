// W8A8 unit tests: the scale and rounding rules in quant.hpp, and the int8
// overload of linear() checked exactly on a hand-computed case and loosely
// against the float projection.
#include "inference_engine/operator/linear.hpp"
#include "inference_engine/quant.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace inference_engine {

namespace {

/// Deterministic values in [-1, 1) that are not symmetric or periodic.
Tensor<float> pattern(std::size_t rows, std::size_t columns,
                      std::size_t offset) {
    std::vector<float> values(rows * columns);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] =
            (static_cast<float>(((i * 37) + (offset * 11)) % 101) / 50.5F) -
            1.0F;
    }
    return Tensor<float>{{rows, columns}, std::move(values)};
}

} // namespace

// The scale maps the largest magnitude to 127, and an all-zero input gets
// scale 1.
TEST(QuantTest, ScaleMapsTheLargestMagnitudeOnto127) {
    EXPECT_EQ(int8_scale(254.0F), 2.0F);
    EXPECT_EQ(int8_scale(127.0F), 1.0F);
    EXPECT_EQ(int8_scale(0.0F), 1.0F);
}

// Quantization rounds ties to even and clamps to the symmetric range [-127,
// 127].
TEST(QuantTest, RoundsTiesToEvenAndClamps) {
    EXPECT_EQ(quantize_int8(2.5F, 1.0F), 2);
    EXPECT_EQ(quantize_int8(3.5F, 1.0F), 4);
    EXPECT_EQ(quantize_int8(-2.5F, 1.0F), -2);
    EXPECT_EQ(quantize_int8(2.4F, 1.0F), 2);
    EXPECT_EQ(quantize_int8(500.0F, 1.0F), 127);
    EXPECT_EQ(quantize_int8(-500.0F, 1.0F), -127);
}

// Each row of a matrix is quantized with a scale from that row alone.
TEST(QuantTest, QuantizesEachRowWithItsOwnScale) {
    const Tensor<float> matrix{
        {3, 3},
        {254.0F, 127.0F, -127.0F, 0.0F, 0.0F, 0.0F, 1.0F, -0.5F, 0.25F}};

    const QuantizedWeight<float> quantized = quantize_rows(matrix);

    ASSERT_EQ(quantized.values.shape, matrix.shape);
    // 127 / 2 = 63.5 is an exact tie and rounds to the even 64.
    const std::vector<std::int8_t> expected_first{127, 64, -64};
    EXPECT_TRUE(std::equal(expected_first.begin(), expected_first.end(),
                           quantized.values.values.begin()));
    EXPECT_EQ(quantized.scales[0], 2.0F);
    // An all-zero row keeps scale 1 rather than dividing by zero.
    EXPECT_EQ(quantized.scales[1], 1.0F);
    EXPECT_EQ(quantized.values.values[3], 0);
    EXPECT_EQ(quantized.scales[2], 1.0F / 127.0F);
    EXPECT_EQ(quantized.values.values[6], 127);
}

// The int8 linear() reproduces a hand-computed W8A8 result exactly, bias
// included.
TEST(QuantTest, LinearMatchesAHandComputedW8A8Case) {
    // The input row's largest magnitude is 127, so its scale is exactly 1
    // and it quantizes to {127, -2, 4} (-2.5 and 3.5 are ties to even).
    const Tensor<float> input{{1, 3}, {127.0F, -2.5F, 3.5F}};
    const Tensor<float> weight{{2, 3},
                               {1.0F, 0.0F, -1.0F, 254.0F, 127.0F, -127.0F}};
    const Tensor<float> bias{{2}, {0.5F, -1.0F}};

    const Tensor<float> output =
        linear<float>(input, quantize_rows(weight), std::cref(bias));

    ASSERT_EQ(output.shape, std::vector<std::size_t>({1, 2}));
    // Row 0 of the weight is {127, 0, -127} at scale 1/127, row 1 is
    // {127, 64, -64} at scale 2.
    const float first = 15621.0F * (1.0F * (1.0F / 127.0F));
    EXPECT_EQ(output.values[0], first + 0.5F);
    EXPECT_EQ(output.values[1], (15745.0F * (1.0F * 2.0F)) - 1.0F);
}

// The int8 linear() stays within 2% of the float projection on patterned data.
TEST(QuantTest, LinearStaysCloseToTheFloatProjection) {
    const Tensor<float> input = pattern(4, 64, 1);
    const Tensor<float> weight = pattern(48, 64, 2);

    const Tensor<float> exact = linear(input, weight);
    const Tensor<float> quantized = linear(input, quantize_rows(weight));

    float largest = 0.0F;
    float worst = 0.0F;
    for (std::size_t i = 0; i < exact.values.size(); ++i) {
        largest = std::max(largest, std::fabs(exact.values[i]));
        worst =
            std::max(worst, std::fabs(exact.values[i] - quantized.values[i]));
    }
    EXPECT_LT(worst, 0.02F * largest);
}

// Infinite and NaN values have no int8 code, so quantizing one throws.
TEST(QuantTest, RejectsNonFiniteValues) {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    EXPECT_THROW(quantize_int8(nan, 1.0F), std::invalid_argument);
    EXPECT_THROW(quantize_int8(1.0F, inf), std::invalid_argument);
    EXPECT_THROW(quantize_rows(Tensor<float>{{1, 2}, {inf, 1.0F}}),
                 std::invalid_argument);
    EXPECT_THROW(quantize_rows(Tensor<float>{{1, 2}, {nan, 1.0F}}),
                 std::invalid_argument);
}

// The int8 linear() rejects a weight whose scale count differs from its row
// count.
TEST(QuantTest, LinearRejectsScalesThatDoNotMatchTheRows) {
    QuantizedWeight<float> weight = quantize_rows(pattern(2, 3, 0));
    weight.scales.pop_back();

    EXPECT_THROW(linear(pattern(1, 3, 1), weight), std::invalid_argument);
}

} // namespace inference_engine
