// Integer-only operator tests (int8.hpp): the rounding and saturation rules
// exactly, each operator on a hand-computed case, and the table-driven ones
// loosely against their float references.
#include "inference_engine/int8.hpp"
#include "inference_engine/operator/activation.hpp"
#include "inference_engine/operator/attention.hpp"
#include "inference_engine/operator/layernorm.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

namespace inference_engine {

namespace {

/// Deterministic int8 values spread over the whole range.
std::vector<std::int8_t> pattern(std::size_t count, std::size_t offset) {
    std::vector<std::int8_t> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = static_cast<std::int8_t>(
            static_cast<int>(((i * 97) + (offset * 31)) % 256) - 128);
    }
    return values;
}

} // namespace

// Half rounds up for both signs, as vnclip does with vxrm = RNU.
TEST(Int8Test, RoundingShiftRoundsHalfUp) {
    EXPECT_EQ(rounding_shift(3, 1), 2);   // 1.5
    EXPECT_EQ(rounding_shift(-3, 1), -1); // -1.5
    EXPECT_EQ(rounding_shift(5, 2), 1);   // 1.25
    EXPECT_EQ(rounding_shift(6, 2), 2);   // 1.5
    EXPECT_EQ(rounding_shift(-6, 2), -1); // -1.5
    EXPECT_EQ(rounding_shift(-7, 2), -2); // -1.75
    EXPECT_EQ(rounding_shift(-7, 0), -7);
}

TEST(Int8Test, SaturatesToTheInt8Range) {
    EXPECT_EQ(saturate_int8(127), 127);
    EXPECT_EQ(saturate_int8(128), 127);
    EXPECT_EQ(saturate_int8(-128), -128);
    EXPECT_EQ(saturate_int8(-100000), -128);
}

TEST(Int8Test, IsqrtIsTheFloorOfTheSquareRoot) {
    std::optional<std::uint32_t> first_wrong;
    for (std::uint32_t value = 0; value < 5000 && !first_wrong; ++value) {
        const std::uint32_t root = isqrt(value);
        if (root * root > value || (root + 1) * (root + 1) <= value) {
            first_wrong = value;
        }
    }
    EXPECT_FALSE(first_wrong.has_value())
        << "isqrt(" << first_wrong.value_or(0) << ") is wrong";
    EXPECT_EQ(isqrt(4294967295U), 65535U);
}

TEST(Int8Test, FloorDivRoundsTowardMinusInfinity) {
    EXPECT_EQ(floor_div(7, 2), 3);
    EXPECT_EQ(floor_div(-7, 2), -4);
    EXPECT_EQ(floor_div(-8, 2), -4);
}

// One output: (1*3 + 2*-4 + 100) = 95, then 95 >> 2 rounds 23.75 to 24.
TEST(Int8Test, LinearSumsAddsBiasThenShifts) {
    const Tensor<std::int8_t> input{{1, 2}, {1, 2}};
    const Tensor<std::int8_t> weight{{2, 2}, {3, -4, 127, 127}};
    const Tensor<std::int32_t> bias{{2}, {100, 0}};

    const Tensor<std::int8_t> output =
        linear_int8(input, weight, 2, std::cref(bias));

    ASSERT_EQ(output.shape, std::vector<std::size_t>({1, 2}));
    EXPECT_EQ(output.values[0], 24);
    EXPECT_EQ(output.values[1], 95);                         // 381 / 4 = 95.25
    EXPECT_EQ(linear_int8(input, weight, 0).values[1], 127); // saturates
}

TEST(Int8Test, LinearRejectsABiasOfTheWrongLength) {
    const Tensor<std::int8_t> input{{1, 2}, {1, 2}};
    const Tensor<std::int8_t> weight{{2, 2}, {1, 1, 1, 1}};
    const Tensor<std::int32_t> bias{{3}, {0, 0, 0}};
    EXPECT_THROW(
        static_cast<void>(linear_int8(input, weight, 0, std::cref(bias))),
        std::invalid_argument);
}

// Every table entry is the float GELU of its input, rounded to one step.
TEST(Int8Test, GeluTableFollowsTheFloatGelu) {
    constexpr float kScale = 0.05F;
    const Int8Table table = gelu_table(kScale, kScale);
    for (int i = 0; i < 256; ++i) {
        const float x = static_cast<float>(i - 128) * kScale;
        const float expected =
            0.5F * x *
            (1.0F + std::tanh(0.7978845608F * (x + (0.044715F * x * x * x))));
        EXPECT_NEAR(static_cast<float>(table[static_cast<std::size_t>(i)]) *
                        kScale,
                    expected, (kScale / 2.0F) + 1e-6F)
            << i;
    }
    const Tensor<std::int8_t> output =
        gelu_int8(Tensor<std::int8_t>{{2}, {0, 127}}, table);
    EXPECT_EQ(output.values[0], 0);
    EXPECT_EQ(output.values[1], 127);
}

TEST(Int8Test, ExpTableStartsAt32768AndDecays) {
    const ExpTable table = exp_table();
    EXPECT_EQ(table[0], 32768);
    EXPECT_EQ(table[16], 12055); // 32768 / e
    EXPECT_EQ(table[255], 0);
    for (std::size_t d = 1; d < table.size(); ++d) {
        EXPECT_LE(table[d], table[d - 1]);
    }
}

// Scores 0..48 in steps of 1/16 are 0..3, so the probabilities are
// softmax(0, 1, 2, 3) of 255 and sum to about 255.
TEST(Int8Test, SoftmaxApproximatesTheFloatSoftmax) {
    const Tensor<std::int32_t> scores{{1, 4}, {0, 16, 32, 48}};
    const Tensor<bool> visible{{1, 4}, {true, true, true, true}};

    const Tensor<std::uint8_t> probs =
        masked_softmax_int8(scores, visible, 0, exp_table());

    const std::vector<float> expected = {0.0321F, 0.0871F, 0.2369F, 0.6439F};
    for (std::size_t k = 0; k < 4; ++k) {
        EXPECT_NEAR(probs.values[k], expected[k] * 255.0F, 1.0F) << k;
    }
    EXPECT_NEAR(std::accumulate(probs.values.begin(), probs.values.end(), 0),
                255, 2);
}

// Equal visible scores split the weight; masked keys get exactly 0, and a
// row with nothing visible is an error.
TEST(Int8Test, SoftmaxMasksKeys) {
    const Tensor<std::int32_t> scores{{1, 4}, {5, 5, 5, 5}};
    const Tensor<bool> visible{{1, 4}, {true, true, false, false}};

    const Tensor<std::uint8_t> probs =
        masked_softmax_int8(scores, visible, 0, exp_table());

    EXPECT_EQ(probs.values, (std::vector<std::uint8_t>{128, 128, 0, 0}));
    const Tensor<bool> none{{1, 2}, {false, false}};
    EXPECT_THROW(
        static_cast<void>(masked_softmax_int8(
            Tensor<std::int32_t>{{1, 2}, {0, 0}}, none, 0, exp_table())),
        std::invalid_argument);
}

// With q and k at scale 1/16 each, scores are at 1/256; shifting by 4
// gives the table's 1/16 steps, so the result tracks the float head.
TEST(Int8Test, AttentionHeadTracksTheFloatHead) {
    constexpr std::size_t kKeys = 6;
    constexpr std::size_t kWidth = 8;
    constexpr float kScale = 1.0F / 16.0F;
    const Tensor<std::int8_t> query{{2, kWidth}, pattern(2 * kWidth, 1)};
    Tensor<std::int8_t> key{{kKeys, kWidth}, pattern(kKeys * kWidth, 2)};
    const Tensor<std::int8_t> value{{kKeys, kWidth},
                                    pattern(kKeys * kWidth, 3)};
    for (std::int8_t& k : key.values) {
        k = static_cast<std::int8_t>(k / 8); // keep scores moderate
    }
    const Tensor<bool> visible = causal_visibility(2, kKeys, 4, 3);

    const Tensor<std::int8_t> output =
        attend_head_int8(query, key, value, visible, 4, exp_table(), 8);

    const auto real = [](const Tensor<std::int8_t>& t, float scale) {
        Tensor<float> r{t.shape, std::vector<float>(t.values.size())};
        for (std::size_t i = 0; i < t.values.size(); ++i) {
            r.values[i] = static_cast<float>(t.values[i]) * scale;
        }
        return r;
    };
    const Tensor<float> expected = attend_head(
        real(query, kScale), real(key, kScale), real(value, 1.0F), visible);
    for (std::size_t i = 0; i < expected.values.size(); ++i) {
        // probs carry 1/255 steps and the >> 8 rescales by 255/256.
        EXPECT_NEAR(static_cast<float>(output.values[i]), expected.values[i],
                    3.0F)
            << i;
    }
}

// Output at scale 1/16 with unit gains tracks the float LayerNorm.
TEST(Int8Test, LayerNormTracksTheFloatLayerNorm) {
    constexpr std::size_t kWidth = 37;
    const Tensor<std::int8_t> input{{2, kWidth}, pattern(2 * kWidth, 4)};
    const Tensor<std::int8_t> unit_gain{{kWidth},
                                        std::vector<std::int8_t>(kWidth, 32)};
    const Tensor<std::int8_t> zero_shift{{kWidth},
                                         std::vector<std::int8_t>(kWidth, 0)};

    const Tensor<std::int8_t> output =
        layernorm_int8(input, unit_gain, zero_shift);

    Tensor<float> real{input.shape, std::vector<float>(input.values.size())};
    for (std::size_t i = 0; i < input.values.size(); ++i) {
        real.values[i] = input.values[i];
    }
    const Tensor<float> expected = layernorm<float>(
        real, Tensor<float>{{kWidth}, std::vector<float>(kWidth, 1.0F)},
        std::nullopt, 0.0F);
    for (std::size_t i = 0; i < expected.values.size(); ++i) {
        EXPECT_NEAR(static_cast<float>(output.values[i]) / 16.0F,
                    expected.values[i], 0.1F)
            << i;
    }
}

TEST(Int8Test, AddSaturates) {
    const Tensor<std::int8_t> a{{3}, {100, -100, 5}};
    const Tensor<std::int8_t> b{{3}, {100, -100, -7}};
    const Tensor<std::int8_t> sum = add_int8(a, b);
    EXPECT_EQ(sum.values, (std::vector<std::int8_t>{127, -128, -2}));
    EXPECT_THROW(static_cast<void>(add_int8(a, Tensor<std::int8_t>{{1}, {0}})),
                 std::invalid_argument);
}

} // namespace inference_engine
