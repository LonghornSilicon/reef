#include "inference_engine/operator/embedding.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace inference_engine {

namespace {

/// Five rows of width three; row r holds {10r, 10r + 1, 10r + 2}.
Tensor<float> make_table() {
    return Tensor<float>{
        {5, 3}, {0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42}};
}

} // namespace

// Each ID picks its table row, in sequence order, including repeats.
TEST(EmbeddingTest, LooksUpRowsInIdOrder) {
    const TokenIds ids{3, 0, 3};

    const auto result = lookup_token_embeddings(ids, make_table());

    const std::vector<std::size_t> expected_shape{3, 3};
    const std::vector<float> expected_values{30, 31, 32, 0, 1, 2, 30, 31, 32};
    EXPECT_EQ(result.shape, expected_shape);
    EXPECT_EQ(result.values, expected_values);
}

// No tokens gives a zero-row tensor that keeps the table width.
TEST(EmbeddingTest, EmptySequenceGivesZeroRows) {
    const auto result = lookup_token_embeddings(TokenIds{}, make_table());

    EXPECT_EQ(result.shape, (std::vector<std::size_t>{0, 3}));
    EXPECT_TRUE(result.values.empty());
}

// IDs past the last row or below zero are rejected.
TEST(EmbeddingTest, RejectsIdsOutsideVocabulary) {
    EXPECT_THROW(lookup_token_embeddings(TokenIds{5}, make_table()),
                 std::invalid_argument);
    EXPECT_THROW(lookup_token_embeddings(TokenIds{-1}, make_table()),
                 std::invalid_argument);
}

// The embedding table must be two-dimensional.
TEST(EmbeddingTest, RejectsMalformedTable) {
    const Tensor<float> flat{{6}, {0, 1, 2, 3, 4, 5}};

    EXPECT_THROW(lookup_token_embeddings(TokenIds{0}, flat),
                 std::invalid_argument);
}

// Prefill: the first token gets position 0, the next position 1, and so on.
TEST(EmbeddingTest, PrefillAddsPositionsFromZero) {
    const Tensor<float> tokens{{2, 3}, {1, 1, 1, 2, 2, 2}};

    const auto result = add_position_embeddings(tokens, make_table(), 0);

    const std::vector<float> expected_values{1, 2, 3, 12, 13, 14};
    EXPECT_EQ(result.shape, tokens.shape);
    EXPECT_EQ(result.values, expected_values);
}

// Decode: a single new token gets the position at the offset (cache length).
TEST(EmbeddingTest, DecodeAddsPositionAtCachedLength) {
    const Tensor<float> token{{1, 3}, {1, 1, 1}};

    const auto result = add_position_embeddings(token, make_table(), 4);

    const std::vector<float> expected_values{41, 42, 43};
    EXPECT_EQ(result.values, expected_values);
}

// Positions may reach the last table row but not go past it.
TEST(EmbeddingTest, RejectsPositionsPastTableEnd) {
    const Tensor<float> tokens{{2, 3}, {0, 0, 0, 0, 0, 0}};

    EXPECT_NO_THROW(add_position_embeddings(tokens, make_table(), 3));
    EXPECT_THROW(add_position_embeddings(tokens, make_table(), 4),
                 std::invalid_argument);
}

// Token and position embeddings must have the same width.
TEST(EmbeddingTest, RejectsWidthMismatch) {
    const Tensor<float> tokens{{1, 2}, {0, 0}};

    EXPECT_THROW(add_position_embeddings(tokens, make_table(), 0),
                 std::invalid_argument);
}

// Integer sums that land exactly on the int8 limits are allowed.
TEST(EmbeddingTest, AddsIntegerPositionsWhenSumsFit) {
    const Tensor<std::int8_t> tokens{{1, 2}, {100, -100}};
    const Tensor<std::int8_t> positions{{1, 2}, {27, -28}};

    const auto result = add_position_embeddings(tokens, positions, 0);

    const std::vector<std::int8_t> expected_values{127, -128};
    EXPECT_EQ(result.values, expected_values);
}

// Integer sums past the int8 limits throw instead of wrapping.
TEST(EmbeddingTest, RejectsIntegerPositionOverflow) {
    const Tensor<std::int8_t> positions{{1, 1}, {1}};

    EXPECT_THROW(add_position_embeddings(Tensor<std::int8_t>{{1, 1}, {127}},
                                         positions, 0),
                 std::overflow_error);
    EXPECT_THROW(add_position_embeddings(Tensor<std::int8_t>{{1, 1}, {-128}},
                                         Tensor<std::int8_t>{{1, 1}, {-1}}, 0),
                 std::overflow_error);
}

} // namespace inference_engine
