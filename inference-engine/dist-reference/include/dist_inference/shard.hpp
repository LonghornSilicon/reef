#pragma once

/** @file
 *  @brief Splitting tensors across ranks, and the two parallel linear()s.
 *
 * A rank is one chip. Every split here is a contiguous range, rank 0 first,
 * and every combine runs in rank order, so the result never depends on which
 * chip finishes first.
 */

#include "inference_engine/operator/linear.hpp"
#include "inference_engine/quant.hpp"
#include "inference_engine/tensor.hpp"
#include "inference_engine/util.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>

namespace dist_inference {

using inference_engine::QuantizedWeight;
using inference_engine::Tensor;

/// Optional reference to a bias, the form linear() takes.
template <typename Scalar>
using OptionalBias =
    std::optional<std::reference_wrapper<const Tensor<Scalar>>>;

/** A contiguous range [start, start + count). */
struct Range {
    std::size_t start{}; ///< First index.
    std::size_t count{}; ///< Number of indices.
};

/** The part of @p total that @p rank holds when it is split @p ranks ways.
 *
 * Rank r gets [total * r / ranks, total * (r + 1) / ranks): equal parts when
 * @p ranks divides @p total, and parts differing by at most one otherwise.
 *
 * @param total Number of indices to split.
 * @param ranks Number of ranks; at least one.
 * @param rank Zero-based rank, below @p ranks.
 * @return The rank's range.
 */
inline Range shard_range(std::size_t total, std::size_t ranks,
                         std::size_t rank) {
    if (ranks == 0 || rank >= ranks) {
        throw std::invalid_argument("shard_range rank is out of range");
    }
    const std::size_t start = total * rank / ranks;
    return {start, (total * (rank + 1) / ranks) - start};
}

/** Shape of a float weight.
 *
 * @tparam Scalar Tensor value type.
 * @param weight Weight shaped [out_features, in_features].
 * @return Its shape.
 */
template <typename Scalar>
const std::vector<std::size_t>& weight_shape(const Tensor<Scalar>& weight) {
    if (weight.shape.size() != 2) {
        throw std::invalid_argument("weight must be a 2-D matrix");
    }
    return weight.shape;
}

/** Shape of a W8A8 weight.
 *
 * @tparam Scalar Floating-point type of the scales.
 * @param weight int8 weight shaped [out_features, in_features].
 * @return Its shape.
 */
template <typename Scalar>
const std::vector<std::size_t>&
weight_shape(const QuantizedWeight<Scalar>& weight) {
    return weight_shape(weight.values);
}

/** Copy a range of rows out of a matrix.
 *
 * @tparam Value Tensor value type.
 * @param matrix Matrix shaped [rows, columns].
 * @param rows Rows to copy.
 * @return Matrix shaped [rows.count, columns].
 */
template <typename Value>
Tensor<Value> slice_rows(const Tensor<Value>& matrix, Range rows) {
    if (!inference_engine::valid_matrix(matrix) ||
        rows.start + rows.count > matrix.shape[0]) {
        throw std::invalid_argument("slice_rows range is outside the matrix");
    }
    const std::size_t columns = matrix.shape[1];
    const auto first = matrix.values.begin() +
                       static_cast<std::ptrdiff_t>(rows.start * columns);
    return {{rows.count, columns},
            std::vector<Value>(first, first + static_cast<std::ptrdiff_t>(
                                                  rows.count * columns))};
}

/** Copy a range of output rows, and their scales, out of a W8A8 weight.
 *
 * @tparam Scalar Floating-point type of the scales.
 * @param weight int8 weight shaped [out_features, in_features].
 * @param rows Output rows to copy.
 * @return int8 weight shaped [rows.count, in_features].
 */
template <typename Scalar>
QuantizedWeight<Scalar> slice_rows(const QuantizedWeight<Scalar>& weight,
                                   Range rows) {
    if (rows.start + rows.count > weight.scales.size()) {
        throw std::invalid_argument("slice_rows range is outside the weight");
    }
    const auto first =
        weight.scales.begin() + static_cast<std::ptrdiff_t>(rows.start);
    return {slice_rows(weight.values, rows),
            std::vector<Scalar>(
                first, first + static_cast<std::ptrdiff_t>(rows.count))};
}

/** Copy a range of columns out of a matrix.
 *
 * @tparam Value Tensor value type.
 * @param matrix Matrix shaped [rows, columns].
 * @param columns Columns to copy.
 * @return Matrix shaped [rows, columns.count].
 */
template <typename Value>
Tensor<Value> slice_columns(const Tensor<Value>& matrix, Range columns) {
    if (!inference_engine::valid_matrix(matrix) ||
        columns.start + columns.count > matrix.shape[1]) {
        throw std::invalid_argument(
            "slice_columns range is outside the matrix");
    }
    const std::size_t rows = matrix.shape[0];
    const std::size_t width = matrix.shape[1];
    Tensor<Value> slice{{rows, columns.count},
                        std::vector<Value>(rows * columns.count)};
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns.count; ++column) {
            slice.values[(row * columns.count) + column] =
                matrix.values[(row * width) + columns.start + column];
        }
    }
    return slice;
}

/** Copy a range of input columns out of a W8A8 weight.
 *
 * The scales stay those of the full rows: a shard of a quantized weight is
 * cut from it, not quantized again.
 *
 * @tparam Scalar Floating-point type of the scales.
 * @param weight int8 weight shaped [out_features, in_features].
 * @param columns Input columns to copy.
 * @return int8 weight shaped [out_features, columns.count], same scales.
 */
template <typename Scalar>
QuantizedWeight<Scalar> slice_columns(const QuantizedWeight<Scalar>& weight,
                                      Range columns) {
    return {slice_columns(weight.values, columns), weight.scales};
}

/** Copy a range out of a one-dimensional tensor, such as a bias.
 *
 * @tparam Scalar Tensor value type.
 * @param vector Tensor shaped [length].
 * @param range Entries to copy.
 * @return Tensor shaped [range.count].
 */
template <typename Scalar>
Tensor<Scalar> slice_vector(const Tensor<Scalar>& vector, Range range) {
    if (vector.shape.size() != 1 ||
        range.start + range.count > vector.shape[0] ||
        !inference_engine::valid_tensor(vector)) {
        throw std::invalid_argument("slice_vector range is outside the tensor");
    }
    const auto first =
        vector.values.begin() + static_cast<std::ptrdiff_t>(range.start);
    return {{range.count},
            std::vector<Scalar>(
                first, first + static_cast<std::ptrdiff_t>(range.count))};
}

/** Place matrices with equal row counts side by side, in order.
 *
 * This is the host's all-gather: it moves values and computes nothing.
 *
 * @tparam Scalar Tensor value type.
 * @param parts One [rows, width_r] matrix per rank, rank 0 first.
 * @return Matrix shaped [rows, sum of width_r].
 */
template <typename Scalar>
Tensor<Scalar> concat_columns(const std::vector<Tensor<Scalar>>& parts) {
    if (parts.empty()) {
        throw std::invalid_argument("concat_columns needs at least one part");
    }
    const std::size_t rows = parts.front().shape.at(0);
    std::size_t width = 0;
    for (const Tensor<Scalar>& part : parts) {
        if (!inference_engine::valid_matrix(part) || part.shape[0] != rows) {
            throw std::invalid_argument(
                "concat_columns parts must be matrices with equal rows");
        }
        width += part.shape[1];
    }
    Tensor<Scalar> output{{rows, width}, std::vector<Scalar>(rows * width)};
    std::size_t start = 0;
    for (const Tensor<Scalar>& part : parts) {
        const std::size_t part_width = part.shape[1];
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < part_width; ++column) {
                output.values[(row * width) + start + column] =
                    part.values[(row * part_width) + column];
            }
        }
        start += part_width;
    }
    return output;
}

/** Sum equally shaped partials in rank order, then add a bias once.
 *
 * This is the host's all-reduce: ((p0 + p1) + p2) + ..., then + bias.
 *
 * @tparam Scalar Tensor value type.
 * @param partials One [rows, out_features] matrix per rank, rank 0 first.
 * @param bias Optional bias shaped [out_features].
 * @return The sum, shaped like each partial.
 */
template <typename Scalar>
Tensor<Scalar> sum_in_rank_order(const std::vector<Tensor<Scalar>>& partials,
                                 OptionalBias<Scalar> bias = std::nullopt) {
    if (partials.empty()) {
        throw std::invalid_argument("sum_in_rank_order needs a partial");
    }
    Tensor<Scalar> output = partials.front();
    for (std::size_t rank = 1; rank < partials.size(); ++rank) {
        if (partials[rank].shape != output.shape) {
            throw std::invalid_argument(
                "sum_in_rank_order partials must have equal shapes");
        }
        for (std::size_t index = 0; index < output.values.size(); ++index) {
            output.values[index] = inference_engine::safe_add(
                output.values[index], partials[rank].values[index]);
        }
    }
    if (bias.has_value()) {
        const Tensor<Scalar>& values = bias->get();
        const std::size_t width = output.shape.at(1);
        if (values.shape != std::vector<std::size_t>{width}) {
            throw std::invalid_argument(
                "sum_in_rank_order bias must hold one value per column");
        }
        for (std::size_t index = 0; index < output.values.size(); ++index) {
            output.values[index] = inference_engine::safe_add(
                output.values[index], values.values[index % width]);
        }
    }
    return output;
}

/** linear() split by output features across @p ranks chips.
 *
 * Rank r computes the outputs in shard_range(out_features, ranks, r) from the
 * whole input, with their bias, and the host concatenates them. Each output
 * is computed exactly as linear() computes it, so the result is linear()'s
 * bit for bit. In W8A8 every rank quantizes the same whole input row, so the
 * scales match too.
 *
 * @tparam Scalar Floating-point tensor value type.
 * @tparam Weight Tensor<Scalar> or QuantizedWeight<Scalar>, shaped
 *         [out_features, in_features].
 * @param input Input rows shaped [rows, in_features].
 * @param weight Weight of the whole projection.
 * @param ranks Number of chips; at least one.
 * @param bias Optional bias shaped [out_features].
 * @return Projected rows shaped [rows, out_features].
 */
template <typename Scalar, typename Weight>
Tensor<Scalar>
column_parallel_linear(const Tensor<Scalar>& input, const Weight& weight,
                       std::size_t ranks,
                       OptionalBias<Scalar> bias = std::nullopt) {
    if (ranks == 0) {
        throw std::invalid_argument(
            "column_parallel_linear needs at least one rank");
    }
    const std::size_t out_features = weight_shape(weight).at(0);
    std::vector<Tensor<Scalar>> parts;
    for (std::size_t rank = 0; rank < ranks; ++rank) {
        const Range outputs = shard_range(out_features, ranks, rank);
        std::optional<Tensor<Scalar>> bias_slice;
        if (bias.has_value()) {
            bias_slice = slice_vector(bias->get(), outputs);
        }
        parts.push_back(inference_engine::linear<Scalar>(
            input, slice_rows(weight, outputs),
            bias_slice ? OptionalBias<Scalar>(std::cref(*bias_slice))
                       : std::nullopt));
    }
    return concat_columns(parts);
}

/** linear() split by input features across @p ranks chips (Megatron).
 *
 * Rank r projects its slice shard_range(in_features, ranks, r) of every
 * input row through the matching weight columns, with no bias, giving a
 * partial as wide as the whole output. The host sums the partials with
 * sum_in_rank_order(), which adds the bias once. In W8A8 each rank quantizes
 * its own input slice, so each slice gets its own row scale.
 *
 * With one rank this is linear() bit for bit. With more, every output sums
 * in a different order (and in W8A8 quantizes with different scales), so it
 * differs from linear() by rounding only.
 *
 * @tparam Scalar Floating-point tensor value type.
 * @tparam Weight Tensor<Scalar> or QuantizedWeight<Scalar>, shaped
 *         [out_features, in_features].
 * @param input Input rows shaped [rows, in_features].
 * @param weight Weight of the whole projection.
 * @param ranks Number of chips; at least one.
 * @param bias Optional bias shaped [out_features].
 * @return Projected rows shaped [rows, out_features].
 * @throws std::invalid_argument If the input width does not match the weight.
 */
template <typename Scalar, typename Weight>
Tensor<Scalar> row_parallel_linear(const Tensor<Scalar>& input,
                                   const Weight& weight, std::size_t ranks,
                                   OptionalBias<Scalar> bias = std::nullopt) {
    if (ranks == 0) {
        throw std::invalid_argument(
            "row_parallel_linear needs at least one rank");
    }
    const std::size_t in_features = weight_shape(weight).at(1);
    if (!inference_engine::valid_matrix(input) ||
        input.shape[1] != in_features) {
        throw std::invalid_argument(
            "row_parallel_linear input width does not match the weight");
    }
    std::vector<Tensor<Scalar>> partials;
    for (std::size_t rank = 0; rank < ranks; ++rank) {
        const Range inputs = shard_range(in_features, ranks, rank);
        partials.push_back(inference_engine::linear<Scalar>(
            slice_columns(input, inputs), slice_columns(weight, inputs)));
    }
    return sum_in_rank_order(partials, bias);
}

} // namespace dist_inference
