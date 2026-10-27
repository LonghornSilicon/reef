#pragma once

/** @file
 *  @brief Bit-reproducible exp and tanh for the float operators.
 *
 * std::exp and std::tanh differ between C libraries, so no kernel can match
 * them bit for bit. These versions use only IEEE-754 add, subtract,
 * multiply, divide and sign operations, each correctly rounded, plus
 * building a power of two from its exponent field. Any target that performs
 * the same steps in the same order produces the same bits.
 *
 * The steps must not be fused into multiply-adds, so everything that
 * includes this header is built with -ffp-contract=off (see CMakeLists.txt).
 */

#include <cmath>
#include <cstdint>
#include <cstring>

namespace inference_engine {

/** Compute e^x for x <= 0.
 *
 * Splits x = k ln2 + r with |r| <= ln2 / 2, evaluates e^r with the Cephes
 * expf polynomial, then scales by 2^k. Every softmax argument (score minus
 * row maximum) and every exponent in tanh_float() is at most zero, so the
 * positive half-line is never needed. Inputs below -87 return 0, which keeps
 * 2^k a normal float.
 *
 * @param x Exponent, at most 0.
 * @return e^x, within 1 ulp of the correctly rounded value (checked for
 * every float in [-87, 0]).
 */
inline float exp_nonpositive(float x) {
    constexpr float kMinInput = -87.0F;
    constexpr float kLog2e = 1.44269504088896341F;
    // Adding then subtracting 1.5 * 2^23 rounds to the nearest integer,
    // ties to even, using float adds alone.
    constexpr float kRoundToInteger = 12582912.0F;
    // ln2 split so that k * kLn2High is exact for |k| < 2^15.
    constexpr float kLn2High = 0.693359375F;
    constexpr float kLn2Low = -2.12194440e-4F;
    constexpr std::int32_t kExponentBias = 127;
    constexpr std::uint32_t kMantissaBits = 23;

    if (x < kMinInput) {
        return 0.0F;
    }
    const float k = ((x * kLog2e) + kRoundToInteger) - kRoundToInteger;
    const float r = (x - (k * kLn2High)) - (k * kLn2Low);
    const float r2 = r * r;
    float poly = 1.9875691500E-4F;
    poly = (poly * r) + 1.3981999507E-3F;
    poly = (poly * r) + 8.3334519073E-3F;
    poly = (poly * r) + 4.1665795894E-2F;
    poly = (poly * r) + 1.6666665459E-1F;
    poly = (poly * r) + 5.0000001201E-1F;
    poly = ((poly * r2) + r) + 1.0F;

    // k lies in [-126, 0], so 2^k is a normal float built from its exponent.
    const auto exponent =
        static_cast<std::uint32_t>(static_cast<std::int32_t>(k) + kExponentBias)
        << kMantissaBits;
    float scale = 0.0F;
    std::memcpy(&scale, &exponent, sizeof scale);
    return poly * scale;
}

/** Compute tanh(x) from exp_nonpositive().
 *
 * Uses tanh(|x|) = (1 - e) / (1 + e) with e = exp(-2|x|), then restores the
 * sign of x, so -0 maps to -0.
 *
 * @param x Input value.
 * @return tanh(x), within 1e-7 absolute error.
 */
inline float tanh_float(float x) {
    const float e = exp_nonpositive(-2.0F * std::fabs(x));
    const float magnitude = (1.0F - e) / (1.0F + e);
    return std::copysign(magnitude, x);
}

} // namespace inference_engine
