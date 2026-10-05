#pragma once

/** @file
 *  @brief Token and position embedding interfaces.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "inference_engine/tensor.hpp"

namespace inference_engine {

/** Ordered token IDs for a single sequence, one ID per token position. */
using TokenIds = std::vector<std::int32_t>;

/** Look up one embedding vector for each token ID.
 *
 * @tparam Scalar Tensor value type.
 * @param ids Token IDs in sequence order.
 * @param table Vocabulary-by-width embedding table.
 * @return Sequence-by-width embedding tensor.
 * @throws std::invalid_argument If the table is malformed, the sequence is
 * too long, or a token ID is outside the vocabulary.
 */
template <typename Scalar>
Tensor<Scalar> lookup_token_embeddings(const TokenIds& ids,
                                       const Tensor<Scalar>& table) {
    if (!detail::valid_matrix(table)) {
        throw std::invalid_argument(
            "lookup_token_embeddings requires a valid 2D embedding table!");
    }

    const std::size_t vocabulary = table.shape[0];
    const std::size_t width = table.shape[1];
    const std::size_t length = ids.size();

    if (length > std::numeric_limits<std::size_t>::max() / width) {
        throw std::invalid_argument(
            "lookup_token_embeddings sequence is too long!");
    }

    Tensor<Scalar> result{{length, width}, std::vector<Scalar>(length * width)};
    for (std::size_t position = 0; position < length; ++position) {
        const std::int32_t id = ids[position];
        if (id < 0 || static_cast<std::size_t>(id) >= vocabulary) {
            throw std::invalid_argument(
                "lookup_token_embeddings token ID is outside the vocabulary!");
        }
        // iterate over rows
        const std::size_t row = static_cast<std::size_t>(id) * width;
        for (std::size_t c = 0; c < width; ++c) {
            result.values[(position * width) + c] = table.values[row + c];
        }
    }

    return result;
}

/** Add learned position vectors to token embeddings.
 *
 * @tparam Scalar Arithmetic value type. Integer sums must fit in Scalar.
 * @param token_embeddings Sequence-by-width token values.
 * @param position_table Position-by-width embedding table.
 * @param position_offset Position of the first input token.
 * @return Sequence-by-width tensor with position values added.
 * @throws std::invalid_argument If shapes are incompatible or positions run
 * past the end of the table.
 * @throws std::overflow_error If an integer sum cannot be represented.
 */
template <typename Scalar>
Tensor<Scalar> add_position_embeddings(const Tensor<Scalar>& token_embeddings,
                                       const Tensor<Scalar>& position_table,
                                       std::size_t position_offset) {
    if (!detail::valid_matrix(position_table) ||
        token_embeddings.shape.size() != 2 ||
        token_embeddings.shape[1] != position_table.shape[1] ||
        token_embeddings.values.size() !=
            token_embeddings.shape[0] * token_embeddings.shape[1]) {
        throw std::invalid_argument(
            "add_position_embeddings requires 2D inputs of equal width!");
    }

    const std::size_t length = token_embeddings.shape[0];
    const std::size_t width = token_embeddings.shape[1];
    const std::size_t max_positions = position_table.shape[0];

    if (position_offset > max_positions ||
        length > max_positions - position_offset) {
        throw std::invalid_argument(
            "add_position_embeddings exceeds the maximum position!");
    }

    Tensor<Scalar> result = token_embeddings;
    for (std::size_t position = 0; position < length; ++position) {
        const std::size_t table_row = (position_offset + position) * width;
        const std::size_t result_row = position * width;
        for (std::size_t column = 0; column < width; ++column) {
            Scalar& value = result.values[result_row + column];
            const Scalar position_value =
                position_table.values[table_row + column];
            if constexpr (std::is_integral_v<Scalar>) {
                // Integer sums can wrap silently, so fail loudly instead.
                if ((position_value > 0 &&
                     value >
                         std::numeric_limits<Scalar>::max() - position_value) ||
                    (position_value < 0 &&
                     value <
                         std::numeric_limits<Scalar>::min() - position_value)) {
                    throw std::overflow_error(
                        "add_position_embeddings sum overflow!");
                }
            }
            value += position_value;
        }
    }
    return result;
}

} // namespace inference_engine
