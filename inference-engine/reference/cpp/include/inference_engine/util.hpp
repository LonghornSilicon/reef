#pragma once

/** @file
 *  @brief Checked arithmetic shared across operators.
 */

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace inference_engine {

/** Internal helpers for the checked arithmetic utilities. */
namespace detail {

/** Dependent false for static_assert in branches that must not instantiate.
 *
 * @tparam Unsupported Type that reached an unsupported branch.
 */
template <typename Unsupported> inline constexpr bool kAlwaysFalse = false;

/** Whether an integer sum would leave Scalar's range.
 *
 * @tparam Scalar Integral value type.
 * @param left First addend.
 * @param right Second addend.
 * @return Whether left + right overflows or underflows Scalar.
 */
template <typename Scalar> bool add_overflows(Scalar left, Scalar right) {
    constexpr auto kMax = std::numeric_limits<Scalar>::max();
    constexpr auto kMin = std::numeric_limits<Scalar>::min();
    if constexpr (std::is_signed_v<Scalar>) {
        return right > 0 ? left > kMax - right
                         : right < 0 && left < kMin - right;
    } else {
        return left > kMax - right;
    }
}

/** Whether an integer product would leave Scalar's range.
 *
 * @tparam Scalar Integral value type.
 * @param left Left factor.
 * @param right Right factor.
 * @return Whether left * right overflows or underflows Scalar.
 */
template <typename Scalar> bool multiply_overflows(Scalar left, Scalar right) {
    constexpr auto kMax = std::numeric_limits<Scalar>::max();
    constexpr auto kMin = std::numeric_limits<Scalar>::min();
    if constexpr (std::is_signed_v<Scalar>) {
        if (left > 0) {
            return right > 0 ? left > kMax / right : right < kMin / left;
        }
        if (left < 0) {
            return right > 0 ? left < kMin / right
                             : right < 0 && right < kMax / left;
        }
        return false;
    } else {
        return left != 0 && right > kMax / left;
    }
}

} // namespace detail

/** Add two values, throwing instead of overflowing.
 *
 * Integral sums that would wrap throw. Floating-point sums that are
 * infinite or NaN throw, whether from overflow or from a non-finite input.
 * Types outside the built-in integer and floating-point ones, including a
 * future custom type such as uint4, fail to compile until support is added
 * explicitly.
 *
 * @tparam Scalar Arithmetic value type.
 * @param left First addend.
 * @param right Second addend.
 * @return left + right.
 * @throws std::overflow_error If the sum cannot be represented in Scalar.
 */
template <typename Scalar> Scalar safe_add(Scalar left, Scalar right) {
    if constexpr (std::is_integral_v<Scalar> && !std::is_same_v<Scalar, bool>) {
        if (detail::add_overflows(left, right)) {
            throw std::overflow_error("safe_add sum overflow");
        }
        return static_cast<Scalar>(left + right);
    } else if constexpr (std::is_floating_point_v<Scalar>) {
        const Scalar sum = left + right;
        if (std::isinf(sum) || std::isnan(sum)) {
            throw std::overflow_error("safe_add sum overflow");
        }
        return sum;
    } else {
        static_assert(detail::kAlwaysFalse<Scalar>,
                      "safe_add supports built-in integer and floating-point "
                      "types only");
    }
}

/** Multiply two values, throwing instead of overflowing.
 *
 * Integral products that would wrap throw. Floating-point products that are
 * infinite or NaN throw, whether from overflow or from a non-finite input.
 * Types outside the built-in integer and floating-point ones, including a
 * future custom type such as uint4, fail to compile until support is added
 * explicitly.
 *
 * @tparam Scalar Arithmetic value type.
 * @param left Left factor.
 * @param right Right factor.
 * @return left * right.
 * @throws std::overflow_error If the product cannot be represented in Scalar.
 */
template <typename Scalar> Scalar safe_multiply(Scalar left, Scalar right) {
    if constexpr (std::is_integral_v<Scalar> && !std::is_same_v<Scalar, bool>) {
        if (detail::multiply_overflows(left, right)) {
            throw std::overflow_error("safe_multiply product overflow");
        }
        return static_cast<Scalar>(left * right);
    } else if constexpr (std::is_floating_point_v<Scalar>) {
        const Scalar product = left * right;
        if (std::isinf(product) || std::isnan(product)) {
            throw std::overflow_error("safe_multiply product overflow");
        }
        return product;
    } else {
        static_assert(detail::kAlwaysFalse<Scalar>,
                      "safe_multiply supports built-in integer and "
                      "floating-point types only");
    }
}

} // namespace inference_engine
