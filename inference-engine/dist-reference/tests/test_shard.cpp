// Shard helpers and the two parallel linear()s: column splits give linear()'s
// bits, row splits sum in rank order with the bias added once.
#include "dist_inference/shard.hpp"

#include "inference_engine/operator/linear.hpp"
#include "inference_engine/quant.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace dist_inference {

namespace {

/// [3, 4] weight with distinct values, so a wrong slice is visible.
Tensor<float> weight_3x4() {
    return Tensor<float>{{3, 4},
                         {1.0F, -0.5F, 0.25F, 2.0F, -1.0F, 0.75F, 0.5F, -0.125F,
                          0.3F, 0.1F, -0.7F, 1.5F}};
}

const Tensor<float> sample_input{
    {2, 4}, {0.1F, -0.2F, 4.0F, -8.0F, 0.6F, 0.3F, -1.1F, 2.2F}};
const Tensor<float> sample_bias{{3}, {0.5F, -1.0F, 2.0F}};

/// True if the ranks' ranges tile [0, total) in rank order, each within one
/// of the others' size.
bool tiles(std::size_t total, std::size_t ranks) {
    std::size_t next = 0;
    for (std::size_t rank = 0; rank < ranks; ++rank) {
        const Range range = shard_range(total, ranks, rank);
        if (range.start != next || range.count < total / ranks ||
            range.count > (total / ranks) + 1) {
            return false;
        }
        next += range.count;
    }
    return next == total;
}

} // namespace

// Ranges are contiguous, rank 0 first, cover every index once, and differ in
// size by at most one.
TEST(ShardTest, ShardRangeCoversEveryIndexOnce) {
    for (const std::size_t total : {0U, 1U, 7U, 8U, 50257U}) {
        for (const std::size_t ranks : {1U, 2U, 3U, 4U}) {
            EXPECT_TRUE(tiles(total, ranks)) << total << ' ' << ranks;
        }
    }
}

// Zero ranks and a rank past the last are rejected.
TEST(ShardTest, ShardRangeRejectsBadRanks) {
    EXPECT_THROW(shard_range(4, 0, 0), std::invalid_argument);
    EXPECT_THROW(shard_range(4, 2, 2), std::invalid_argument);
}

// concat_columns places parts side by side in rank order.
TEST(ShardTest, ConcatColumnsInRankOrder) {
    const Tensor<float> joined = concat_columns<float>(
        {Tensor<float>{{2, 1}, {1.0F, 2.0F}},
         Tensor<float>{{2, 2}, {3.0F, 4.0F, 5.0F, 6.0F}}});

    EXPECT_EQ(joined.shape, std::vector<std::size_t>({2, 3}));
    EXPECT_EQ(joined.values,
              std::vector<float>({1.0F, 3.0F, 4.0F, 2.0F, 5.0F, 6.0F}));
}

// A column split, even an uneven one, is linear() bit for bit, in float and
// in W8A8.
TEST(ShardTest, ColumnParallelIsLinearBitwise) {
    const QuantizedWeight<float> quantized =
        inference_engine::quantize_rows(weight_3x4());
    for (const std::size_t ranks : {1U, 2U, 3U}) {
        EXPECT_EQ(column_parallel_linear<float>(sample_input, weight_3x4(),
                                                ranks, std::cref(sample_bias))
                      .values,
                  inference_engine::linear<float>(sample_input, weight_3x4(),
                                                  std::cref(sample_bias))
                      .values)
            << ranks;
        EXPECT_EQ(column_parallel_linear<float>(sample_input, quantized, ranks,
                                                std::cref(sample_bias))
                      .values,
                  inference_engine::linear<float>(sample_input, quantized,
                                                  std::cref(sample_bias))
                      .values)
            << ranks;
    }
}

// One rank of a row split is linear() bit for bit, in float and in W8A8.
TEST(ShardTest, RowParallelWithOneRankIsLinear) {
    const QuantizedWeight<float> quantized =
        inference_engine::quantize_rows(weight_3x4());

    EXPECT_EQ(row_parallel_linear<float>(sample_input, weight_3x4(), 1,
                                         std::cref(sample_bias))
                  .values,
              inference_engine::linear<float>(sample_input, weight_3x4(),
                                              std::cref(sample_bias))
                  .values);
    EXPECT_EQ(row_parallel_linear<float>(sample_input, quantized, 1,
                                         std::cref(sample_bias))
                  .values,
              inference_engine::linear<float>(sample_input, quantized,
                                              std::cref(sample_bias))
                  .values);
}

// Each rank projects its slice with no bias; the partials are summed and the
// bias is added once, last.
TEST(ShardTest, RowParallelAddsTheBiasOnce) {
    const Tensor<float> input{{1, 4}, {1.0F, 2.0F, 3.0F, 4.0F}};
    const Tensor<float> weight{
        {2, 4}, {1.0F, 1.0F, 1.0F, 1.0F, 0.5F, -1.0F, 2.0F, 0.25F}};
    const Tensor<float> bias{{2}, {10.0F, -10.0F}};

    const Tensor<float> output =
        row_parallel_linear<float>(input, weight, 2, std::cref(bias));

    // Rank 0: [1, 2] . [1, 1] and [0.5, -1]; rank 1: [3, 4] . [1, 1] and
    // [2, 0.25].
    EXPECT_EQ(output.values, std::vector<float>({(3.0F + 7.0F) + 10.0F,
                                                 (-1.5F + 7.0F) + -10.0F}));
}

// Splitting the sum changes only its rounding. In float 1e8 + 1 rounds back
// to 1e8. One chip computes ((1 + 1e8) - 1e8) + 1 = 1: the first 1 is lost,
// the last survives. Two ranks compute (1 + 1e8) + (-1e8 + 1) = 0: both are
// lost.
TEST(ShardTest, RowParallelRoundsDifferently) {
    const Tensor<float> input{{1, 4}, {1.0F, 1e8F, -1e8F, 1.0F}};
    const Tensor<float> ones{{1, 4}, {1.0F, 1.0F, 1.0F, 1.0F}};

    EXPECT_EQ(inference_engine::linear(input, ones).values[0], 1.0F);
    EXPECT_EQ(row_parallel_linear<float>(input, ones, 2).values[0], 0.0F);
}

// The order is fixed, rank 0 first: with partials 1, 1e8 and -1e8,
// (1 + 1e8) - 1e8 = 0, whereas the reverse order would give 1.
TEST(ShardTest, RowParallelSumsInRankOrder) {
    const Tensor<float> input{{1, 3}, {1.0F, 1e8F, -1e8F}};
    const Tensor<float> ones{{1, 3}, {1.0F, 1.0F, 1.0F}};

    EXPECT_EQ(row_parallel_linear<float>(input, ones, 3).values[0], 0.0F);
}

// A W8A8 row shard keeps the full rows' weight scales but quantizes its own
// input slice, so it equals linear() on that slice with the cut weight.
TEST(ShardTest, RowParallelW8A8ShardsQuantizeTheirOwnSlice) {
    const QuantizedWeight<float> quantized =
        inference_engine::quantize_rows(weight_3x4());
    const auto shard = [&](Range inputs) {
        return inference_engine::linear<float>(
            slice_columns(sample_input, inputs),
            slice_columns(quantized, inputs));
    };

    const Tensor<float> output =
        row_parallel_linear<float>(sample_input, quantized, 2);

    const Tensor<float> first = shard({0, 2});
    const Tensor<float> second = shard({2, 2});
    for (std::size_t index = 0; index < output.values.size(); ++index) {
        EXPECT_EQ(output.values[index],
                  first.values[index] + second.values[index])
            << index;
    }
    EXPECT_EQ(slice_columns(quantized, {2, 2}).scales, quantized.scales);
}

// Zero ranks and an input narrower than the weight are rejected.
TEST(ShardTest, ParallelLinearsRejectBadArguments) {
    const Tensor<float> narrow{{1, 2}, {1.0F, 2.0F}};

    EXPECT_THROW(row_parallel_linear<float>(sample_input, weight_3x4(), 0),
                 std::invalid_argument);
    EXPECT_THROW(column_parallel_linear<float>(sample_input, weight_3x4(), 0),
                 std::invalid_argument);
    EXPECT_THROW(row_parallel_linear<float>(narrow, weight_3x4(), 2),
                 std::invalid_argument);
}

} // namespace dist_inference
