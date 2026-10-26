// exp_nonpositive() and tanh_float() unit tests: accuracy against the C
// library in double precision, and the exact edge cases the Coral kernels
// must reproduce.
#include "inference_engine/math.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace inference_engine {

namespace {

std::int64_t float_bits(float value) {
    std::int32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
}

float bits_float(std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

} // namespace

// exp_nonpositive() is within one ulp of the correctly rounded e^x across [-87,
// 0].
TEST(MathTest, ExpIsWithinOneUlpAcrossItsDomain) {
    // Negative floats sort by bit pattern, so stepping the pattern sweeps
    // [-87, -0] evenly in ulps. Every float was checked once; this samples.
    const auto first = static_cast<std::uint32_t>(float_bits(-0.0F));
    const auto last = static_cast<std::uint32_t>(float_bits(-87.0F));
    constexpr std::uint32_t kStride = 997;
    std::int64_t worst = 0;
    for (std::uint32_t bits = first; bits <= last; bits += kStride) {
        const float x = bits_float(bits);
        const auto exact = static_cast<float>(std::exp(static_cast<double>(x)));
        worst = std::max(worst, std::llabs(float_bits(exp_nonpositive(x)) -
                                           float_bits(exact)));
    }
    EXPECT_LE(worst, 1);
}

// exp_nonpositive() is exact at zero, flushes below -87 to zero, and stays
// positive at -87.
TEST(MathTest, ExpHandlesItsEdges) {
    EXPECT_EQ(exp_nonpositive(0.0F), 1.0F);
    EXPECT_EQ(exp_nonpositive(-0.0F), 1.0F);
    EXPECT_EQ(exp_nonpositive(-87.5F), 0.0F);
    EXPECT_EQ(exp_nonpositive(-1000.0F), 0.0F);
    EXPECT_GT(exp_nonpositive(-87.0F), 0.0F);
    EXPECT_NEAR(exp_nonpositive(-1.0F), 0.36787944F, 1e-7F);
}

// tanh_float() is within 1e-7 of tanh over [-12, 12] and exactly odd.
TEST(MathTest, TanhIsAccurateAndOdd) {
    // Steps of 1/1024 over [-12, 12].
    constexpr int kSteps = 12 * 1024;
    for (int step = -kSteps; step <= kSteps; ++step) {
        const float x = static_cast<float>(step) / 1024.0F;
        EXPECT_NEAR(tanh_float(x), std::tanh(static_cast<double>(x)), 1e-7)
            << x;
        EXPECT_EQ(tanh_float(-x), -tanh_float(x)) << x;
    }
}

// tanh_float() keeps the sign of zero and saturates to exactly +-1.
TEST(MathTest, TanhKeepsSignedZeroAndSaturates) {
    EXPECT_EQ(tanh_float(0.0F), 0.0F);
    EXPECT_FALSE(std::signbit(tanh_float(0.0F)));
    EXPECT_TRUE(std::signbit(tanh_float(-0.0F)));
    EXPECT_EQ(tanh_float(50.0F), 1.0F);
    EXPECT_EQ(tanh_float(-50.0F), -1.0F);
}

} // namespace inference_engine
