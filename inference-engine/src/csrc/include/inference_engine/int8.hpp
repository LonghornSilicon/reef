#pragma once

/** @file
 *  @brief Integer-only operators: int8 activations, int32 sums, no floats.
 *
 * Every scale change is a rounding right shift or a lookup in a 256-entry
 * table built on the host, so a core without float units (or that avoids
 * them) can run these. Integer results do not depend on summation order,
 * so the Coral *_int8 kernels reproduce these functions exactly.
 *
 * Rounding follows the RVV narrowing instructions with vxrm = RNU: shift
 * right, then add the last bit shifted out (round half up), then saturate.
 */

#include "inference_engine/math.hpp"
#include "inference_engine/operator/activation.hpp"
#include "inference_engine/tensor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace inference_engine {

/// A 256-entry int8 table indexed by an int8 value plus 128.
using Int8Table = std::array<std::int8_t, 256>;
/// Softmax weights by score distance below the row maximum (see exp_table()).
using ExpTable = std::array<std::uint16_t, 256>;

/** Shift right, rounding half up, as vnclip does with vxrm = RNU.
 *
 * @param value Value to shift.
 * @param shift Bits to drop, at most 31.
 * @return floor(value / 2^shift + 1/2).
 */
inline std::int32_t rounding_shift(std::int32_t value, unsigned shift) {
    if (shift == 0) {
        return value;
    }
    // >> on a negative int32 is an arithmetic shift on every target we use.
    return (value >> shift) + ((value >> (shift - 1)) & 1);
}

/** Clamp to the int8 range.
 *
 * @param value Value to clamp.
 * @return value limited to [-128, 127].
 */
inline std::int8_t saturate_int8(std::int32_t value) {
    return static_cast<std::int8_t>(
        std::clamp<std::int32_t>(value, std::numeric_limits<std::int8_t>::min(),
                                 std::numeric_limits<std::int8_t>::max()));
}

/** Integer projection: saturate(rounding_shift(input . weight^T + bias)).
 *
 * @param input Rows [rows, in_features].
 * @param weight [out_features, in_features], the layout linear() uses.
 * @param shift Bits dropped from each int32 sum: the output scale is the
 *        input scale times the weight scale times 2^shift.
 * @param bias Optional reference to an int32 bias [out_features] at the
 *        sum's scale, added before the shift.
 * @return Projected rows [rows, out_features].
 * @throws std::invalid_argument If the shapes do not fit together.
 * @throws std::overflow_error If an int32 sum overflows.
 */
inline Tensor<std::int8_t> linear_int8(
    const Tensor<std::int8_t>& input, const Tensor<std::int8_t>& weight,
    unsigned shift,
    std::optional<std::reference_wrapper<const Tensor<std::int32_t>>> bias =
        std::nullopt) {
    if (weight.shape.size() != 2) {
        throw std::invalid_argument("linear_int8 weight must be a 2-D matrix");
    }
    const Tensor<std::int32_t> sums =
        matmul<std::int8_t, std::int32_t>(input, transpose(weight, 0, 1));
    const std::size_t out_features = sums.shape[1];
    if (bias.has_value() &&
        (bias->get().shape != std::vector<std::size_t>{out_features} ||
         !valid_tensor(bias->get()))) {
        throw std::invalid_argument(
            "linear_int8 bias must hold one value per output feature");
    }
    Tensor<std::int8_t> output{sums.shape,
                               std::vector<std::int8_t>(sums.values.size())};
    for (std::size_t index = 0; index < sums.values.size(); ++index) {
        std::int32_t sum = sums.values[index];
        if (bias.has_value()) {
            sum = safe_add(sum, bias->get().values[index % out_features]);
        }
        output.values[index] = saturate_int8(rounding_shift(sum, shift));
    }
    return output;
}

/** GELU as a table: entry i is GELU((i - 128) * in_scale) / out_scale.
 *
 * Uses the float gelu() reference, rounds half away from zero and
 * saturates, so the table is the same on every host.
 *
 * @param in_scale Real value of one input step.
 * @param out_scale Real value of one output step.
 * @return The table gelu_int8() looks up.
 */
inline Int8Table gelu_table(float in_scale, float out_scale) {
    constexpr std::size_t kEntries = 256;
    constexpr int kZero = 128;
    Tensor<float> inputs{{kEntries}, std::vector<float>(kEntries)};
    for (std::size_t i = 0; i < kEntries; ++i) {
        inputs.values[i] =
            static_cast<float>(static_cast<int>(i) - kZero) * in_scale;
    }
    const Tensor<float> outputs = gelu(inputs);
    Int8Table table{};
    for (std::size_t i = 0; i < kEntries; ++i) {
        table[i] = saturate_int8(static_cast<std::int32_t>(
            std::lround(outputs.values[i] / out_scale)));
    }
    return table;
}

/** Apply a 256-entry table elementwise: y = table[x + 128].
 *
 * @param input int8 values.
 * @param table Table from gelu_table() or another int8 activation.
 * @return Looked-up values with the input's shape.
 */
inline Tensor<std::int8_t> gelu_int8(const Tensor<std::int8_t>& input,
                                     const Int8Table& table) {
    constexpr int kZero = 128;
    Tensor<std::int8_t> output{input.shape,
                               std::vector<std::int8_t>(input.values.size())};
    for (std::size_t i = 0; i < input.values.size(); ++i) {
        const int index = static_cast<int>(input.values[i]) + kZero;
        output.values[i] = table[static_cast<std::size_t>(index)];
    }
    return output;
}

/** Softmax weights: entry d is round(32768 * e^(-d * step)).
 *
 * A score d steps below its row maximum gets weight entry d; the maximum
 * itself gets 32768. Built from exp_nonpositive(), so it is the same on
 * every host.
 *
 * @param step Real score distance of one step; 1/16 by default.
 * @return The table masked_softmax_int8() looks up.
 */
inline ExpTable exp_table(float step = 1.0F / 16.0F) {
    constexpr float kOne = 32768.0F;
    ExpTable table{};
    for (std::size_t d = 0; d < table.size(); ++d) {
        table[d] = static_cast<std::uint16_t>(
            std::lround(kOne * exp_nonpositive(-static_cast<float>(d) * step)));
    }
    return table;
}

/** Integer softmax over the last dimension, restricted to visible keys.
 *
 * Per row: d = (max - score) >> shift, e = table[min(d, 255)], then
 * p = round_half_up(e * (255 * 2^16 / sum(e)) / 2^16), saturated to 255.
 * Visible probabilities sum to about 255; masked keys get 0.
 *
 * @param scores int32 scores [rows, keys].
 * @param visible [rows, keys] mask, true where a key participates.
 * @param shift Score bits dropped before the table lookup.
 * @param table Weights from exp_table().
 * @return uint8 probabilities [rows, keys].
 * @throws std::invalid_argument If the mask does not match or a row has no
 * visible key.
 */
inline Tensor<std::uint8_t>
masked_softmax_int8(const Tensor<std::int32_t>& scores,
                    const Tensor<bool>& visible, unsigned shift,
                    const ExpTable& table) {
    if (!valid_matrix(scores) || visible.shape != scores.shape ||
        !valid_tensor(visible)) {
        throw std::invalid_argument(
            "masked_softmax_int8 needs a mask shaped like the scores");
    }
    constexpr std::uint32_t kLastEntry = 255;
    constexpr std::uint32_t kProbabilityOne = 255U << 16U;
    const std::size_t rows = scores.shape[0];
    const std::size_t keys = scores.shape[1];
    Tensor<std::uint8_t> probs{scores.shape,
                               std::vector<std::uint8_t>(rows * keys, 0)};
    std::vector<std::uint32_t> weights(keys);
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t start = row * keys;
        std::optional<std::int32_t> largest;
        for (std::size_t k = 0; k < keys; ++k) {
            if (visible.values[start + k]) {
                largest = std::max(largest.value_or(scores.values[start + k]),
                                   scores.values[start + k]);
            }
        }
        if (!largest.has_value()) {
            throw std::invalid_argument(
                "masked_softmax_int8 row has no visible key");
        }
        std::uint32_t sum = 0;
        for (std::size_t k = 0; k < keys; ++k) {
            if (visible.values[start + k]) {
                const auto d = static_cast<std::uint32_t>(
                    (*largest - scores.values[start + k]) >>
                    static_cast<std::int32_t>(shift));
                weights[k] = table[std::min(d, kLastEntry)];
                sum += weights[k];
            }
        }
        // The maximum's weight is 32768, so sum is never zero.
        const std::uint32_t reciprocal = kProbabilityOne / sum;
        for (std::size_t k = 0; k < keys; ++k) {
            if (visible.values[start + k]) {
                const std::uint32_t scaled = weights[k] * reciprocal;
                const std::uint32_t rounded =
                    (scaled >> 16U) + ((scaled >> 15U) & 1U);
                probs.values[start + k] =
                    static_cast<std::uint8_t>(std::min(rounded, kLastEntry));
            }
        }
    }
    return probs;
}

/** Attend one head in integers.
 *
 * scores = query . key^T in int32, probs = masked_softmax_int8(scores),
 * then out = saturate(rounding_shift(probs . value, out_shift)).
 *
 * @param query int8 rows [query_len, head_dim].
 * @param key int8 rows [key_len, head_dim].
 * @param value int8 rows [key_len, head_dim].
 * @param visible [query_len, key_len] mask from causal_visibility().
 * @param score_shift Score bits dropped before the exp table lookup.
 * @param table Weights from exp_table().
 * @param out_shift Bits dropped from each probs . value sum; 8 keeps the
 *        value scale, since the probabilities sum to about 2^8.
 * @return int8 context rows [query_len, head_dim].
 * @throws std::invalid_argument If the shapes disagree.
 */
inline Tensor<std::int8_t> attend_head_int8(
    const Tensor<std::int8_t>& query, const Tensor<std::int8_t>& key,
    const Tensor<std::int8_t>& value, const Tensor<bool>& visible,
    unsigned score_shift, const ExpTable& table, unsigned out_shift) {
    const Tensor<std::int32_t> scores =
        matmul<std::int8_t, std::int32_t>(query, transpose(key, 0, 1));
    const Tensor<std::uint8_t> probs =
        masked_softmax_int8(scores, visible, score_shift, table);
    if (!valid_matrix(value) || value.shape[0] != key.shape[0]) {
        throw std::invalid_argument("attend_head_int8 value shape mismatch");
    }
    const std::size_t rows = query.shape[0];
    const std::size_t keys = value.shape[0];
    const std::size_t width = value.shape[1];
    Tensor<std::int8_t> output{{rows, width},
                               std::vector<std::int8_t>(rows * width)};
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t c = 0; c < width; ++c) {
            std::int32_t sum = 0;
            for (std::size_t k = 0; k < keys; ++k) {
                sum +=
                    static_cast<std::int32_t>(probs.values[(row * keys) + k]) *
                    static_cast<std::int32_t>(value.values[(k * width) + c]);
            }
            output.values[(row * width) + c] =
                saturate_int8(rounding_shift(sum, out_shift));
        }
    }
    return output;
}

/// LayerNorm outputs have scale 2^-kLayerNormOutputBits (1/16): +-8 in int8.
inline constexpr unsigned kLayerNormOutputBits = 4;
/// LayerNorm gains are fixed point with this many fraction bits (32 = 1.0).
inline constexpr unsigned kLayerNormGainBits = 5;

/** Floor of the square root.
 *
 * @param value Non-negative integer.
 * @return The largest r with r * r <= value.
 */
inline std::uint32_t isqrt(std::uint32_t value) {
    std::uint32_t root = 0;
    for (std::uint32_t bit = 1U << 30U; bit != 0; bit >>= 2U) {
        if (value >= root + bit) {
            value -= root + bit;
            root = (root >> 1U) + bit;
        } else {
            root >>= 1U;
        }
    }
    return root;
}

/** Integer division rounding toward minus infinity.
 *
 * @param numerator Dividend.
 * @param denominator Positive divisor.
 * @return floor(numerator / denominator).
 */
inline std::int32_t floor_div(std::int32_t numerator,
                              std::int32_t denominator) {
    const std::int32_t quotient = numerator / denominator;
    return (numerator % denominator != 0 && numerator < 0) ? quotient - 1
                                                           : quotient;
}

/** Integer LayerNorm over the last dimension.
 *
 * Per row: mean = floor(sum / width), std = isqrt(sum((x - mean)^2) /
 * width), inv = 2^(16 + kLayerNormOutputBits) / max(std, 1), then
 *   y = saturate(rounding_shift(((x - mean) * inv >> 16) * gain
 *                               + (shift << kLayerNormGainBits),
 *                               kLayerNormGainBits)).
 * The input scale cancels out; the output scale is 2^-kLayerNormOutputBits.
 *
 * @param input int8 rows [rows, width].
 * @param gain int8 gains [width], kLayerNormGainBits fraction bits.
 * @param shift int8 shifts [width], at the output scale.
 * @return Normalized int8 rows [rows, width].
 * @throws std::invalid_argument If the shapes disagree.
 */
inline Tensor<std::int8_t> layernorm_int8(const Tensor<std::int8_t>& input,
                                          const Tensor<std::int8_t>& gain,
                                          const Tensor<std::int8_t>& shift) {
    if (!valid_matrix(input) ||
        gain.shape != std::vector<std::size_t>{input.shape[1]} ||
        shift.shape != gain.shape || !valid_tensor(gain) ||
        !valid_tensor(shift)) {
        throw std::invalid_argument("layernorm_int8 shape mismatch");
    }
    const std::size_t width = input.shape[1];
    const auto count = static_cast<std::int32_t>(width);
    constexpr std::int32_t kInvOne = 1 << (16U + kLayerNormOutputBits);
    Tensor<std::int8_t> output{input.shape,
                               std::vector<std::int8_t>(input.values.size())};
    for (std::size_t start = 0; start < input.values.size(); start += width) {
        std::int32_t sum = 0;
        for (std::size_t k = 0; k < width; ++k) {
            sum += input.values[start + k];
        }
        const std::int32_t mean = floor_div(sum, count);
        std::int32_t squares = 0;
        for (std::size_t k = 0; k < width; ++k) {
            const std::int32_t centred = input.values[start + k] - mean;
            squares += centred * centred;
        }
        const std::uint32_t deviation =
            isqrt(static_cast<std::uint32_t>(squares / count));
        const std::int32_t inv =
            kInvOne / static_cast<std::int32_t>(std::max(deviation, 1U));
        for (std::size_t k = 0; k < width; ++k) {
            const std::int32_t normed =
                ((input.values[start + k] - mean) * inv) >> 16;
            const std::int32_t scaled =
                (normed * gain.values[k]) +
                (static_cast<std::int32_t>(shift.values[k]) *
                 (1 << kLayerNormGainBits));
            output.values[start + k] =
                saturate_int8(rounding_shift(scaled, kLayerNormGainBits));
        }
    }
    return output;
}

/** Saturating elementwise add of two int8 tensors at the same scale.
 *
 * @param left First addend.
 * @param right Second addend, shaped like @p left.
 * @return saturate(left + right).
 * @throws std::invalid_argument If the shapes differ.
 */
inline Tensor<std::int8_t> add_int8(const Tensor<std::int8_t>& left,
                                    const Tensor<std::int8_t>& right) {
    if (left.shape != right.shape || !valid_tensor(left) ||
        !valid_tensor(right)) {
        throw std::invalid_argument("add_int8 shapes differ");
    }
    Tensor<std::int8_t> output{left.shape,
                               std::vector<std::int8_t>(left.values.size())};
    for (std::size_t i = 0; i < left.values.size(); ++i) {
        output.values[i] = saturate_int8(
            static_cast<std::int32_t>(left.values[i]) + right.values[i]);
    }
    return output;
}

} // namespace inference_engine
