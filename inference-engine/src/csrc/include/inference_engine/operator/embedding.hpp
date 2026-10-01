#pragma once

/** @file
 *  @brief Token and position embedding interfaces.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include "inference_engine/tensor.hpp"

namespace inference_engine {

/** Ordered token IDs for a single sequence, one ID per token position. */
using TokenIds = std::vector<std::int32_t>;

/** Look up one embedding vector for each token ID.
 *
 * @param ids Token IDs in sequence order.
 * @param table Vocabulary-by-width embedding table.
 * @return Sequence-by-width embedding tensor.
 */
// @ Tokenizer team
// TODO: Map token IDs to rows of the token-embedding table. Specify ID bounds,
// output shape, and the table layout to match workloads.Embedding.
Tensor<float> lookup_token_embeddings(const TokenIds& ids,
                                      const Tensor<float>& table);

/** Add learned position vectors to token embeddings.
 *
 * @param token_embeddings Sequence-by-width token values.
 * @param position_table Position-by-width embedding table.
 * @param position_offset Position of the first input token.
 * @return Sequence-by-width tensor with position values added.
 */
// @ Tokenizer team
// TODO: Add learned position embeddings at the correct position offset for
// prefill and single-token decode; check maximum position bounds.
Tensor<float> add_position_embeddings(const Tensor<float>& token_embeddings,
                                      const Tensor<float>& position_table,
                                      std::size_t position_offset);

} // namespace inference_engine
