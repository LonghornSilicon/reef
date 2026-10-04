#include "inference_engine/operator/embedding.hpp"

#include <gtest/gtest.h>

#include <cstddef>
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

TEST(EmbeddingTest, LooksUpRowsInIdOrder) {
    const TokenIds ids{3, 0, 3};

    const auto result = lookup_token_embeddings(ids, make_table());

    const std::vector<std::size_t> expected_shape{3, 3};
    const std::vector<float> expected_values{30, 31, 32, 0, 1, 2, 30, 31, 32};
    EXPECT_EQ(result.shape, expected_shape);
    EXPECT_EQ(result.values, expected_values);
}

TEST(EmbeddingTest, EmptySequenceGivesZeroRows) {
    const auto result = lookup_token_embeddings(TokenIds{}, make_table());

    EXPECT_EQ(result.shape, (std::vector<std::size_t>{0, 3}));
    EXPECT_TRUE(result.values.empty());
}

TEST(EmbeddingTest, RejectsIdsOutsideVocabulary) {
    EXPECT_THROW(lookup_token_embeddings(TokenIds{5}, make_table()),
                 std::invalid_argument);
    EXPECT_THROW(lookup_token_embeddings(TokenIds{-1}, make_table()),
                 std::invalid_argument);
}

TEST(EmbeddingTest, RejectsMalformedTable) {
    const Tensor<float> flat{{6}, {0, 1, 2, 3, 4, 5}};

    EXPECT_THROW(lookup_token_embeddings(TokenIds{0}, flat),
                 std::invalid_argument);
}

TEST(EmbeddingTest, PrefillAddsPositionsFromZero) {
    const Tensor<float> tokens{{2, 3}, {1, 1, 1, 2, 2, 2}};

    const auto result = add_position_embeddings(tokens, make_table(), 0);

    const std::vector<float> expected_values{1, 2, 3, 12, 13, 14};
    EXPECT_EQ(result.shape, tokens.shape);
    EXPECT_EQ(result.values, expected_values);
}

TEST(EmbeddingTest, DecodeAddsPositionAtCachedLength) {
    const Tensor<float> token{{1, 3}, {1, 1, 1}};

    const auto result = add_position_embeddings(token, make_table(), 4);

    const std::vector<float> expected_values{41, 42, 43};
    EXPECT_EQ(result.values, expected_values);
}

TEST(EmbeddingTest, RejectsPositionsPastTableEnd) {
    const Tensor<float> tokens{{2, 3}, {0, 0, 0, 0, 0, 0}};

    EXPECT_NO_THROW(add_position_embeddings(tokens, make_table(), 3));
    EXPECT_THROW(add_position_embeddings(tokens, make_table(), 4),
                 std::invalid_argument);
}

TEST(EmbeddingTest, RejectsWidthMismatch) {
    const Tensor<float> tokens{{1, 2}, {0, 0}};

    EXPECT_THROW(add_position_embeddings(tokens, make_table(), 0),
                 std::invalid_argument);
}

} // namespace inference_engine