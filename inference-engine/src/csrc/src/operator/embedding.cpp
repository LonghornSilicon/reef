#include "inference_engine/operator/embedding.hpp"

// @ Tokenizer team
// TODO: Implement lookup_token_embeddings and add_position_embeddings, then
// compare fixed token IDs and positions with workloads reference outputs.
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace inference_engine {

namespace {
// Helper function just needed for this file

bool valid_table(const Tensor<float>& table) {
    return table.shape.size() == 2 && table.shape[0] > 0 && table.shape[1] > 0 &&
           table.shape[0] <=
               std::numeric_limits<std::size_t>::max() / table.shape[1] &&
           table.values.size() == table.shape[0] * table.shape[1];
}

} // namespace

Tensor<float> lookup_token_embeddings(const TokenIds& ids,
                                      const Tensor<float>& table) {
    if (!valid_table(table)) {
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

    Tensor<float> result{{length, width}, std::vector<float>(length * width)};
    for (std::size_t position = 0; position < length; ++position) {
        const int32_t id = ids[position];
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

Tensor<float> add_position_embeddings(const Tensor<float>& token_embeddings,
                                      const Tensor<float>& position_table,
                                      std::size_t position_offset) {
    if (!valid_table(position_table) || token_embeddings.shape.size() != 2 ||
        token_embeddings.shape[1] != position_table.shape[1] ||
        token_embeddings.values.size() !=
            token_embeddings.shape[0] * token_embeddings.shape[1]) {
        throw std::invalid_argument(
            "add_position_embedding requires 2D inputs of equal width!");
    }

    const std::size_t length = token_embeddings.shape[0];
    const std::size_t width = token_embeddings.shape[1];
    const std::size_t max_positions = position_table.shape[0];

    if (position_offset > max_positions ||
        length > max_positions - position_offset) {
        throw std::invalid_argument(
            "add_position_embeddings exceeds the maximum position!");
    }

    Tensor<float> result = token_embeddings;
    for (std::size_t position = 0; position < length; ++position) {
        const std::size_t table_row = (position_offset + position) * width;
        const std::size_t result_row = position * width;
        for (std::size_t column = 0; column < width; ++column) {
            result.values[result_row + column] +=
                position_table.values[table_row + column];
        }
    }
    return result;
}

} // namespace inference_engine