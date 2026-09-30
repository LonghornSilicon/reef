#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "inference_engine/tensor.hpp"

namespace inference_engine {

using TokenIds = std::vector<std::int32_t>;

// @ Tokenizer team
// TODO: Map token IDs to rows of the token-embedding table. Specify ID bounds,
// output shape, and the table layout to match workloads.Embedding.
Tensor lookup_token_embeddings(const TokenIds& ids, const Tensor& table);

// @ Tokenizer team
// TODO: Add learned position embeddings at the correct position offset for
// prefill and single-token decode; check maximum position bounds.
Tensor add_position_embeddings(const Tensor& token_embeddings,
                               const Tensor& position_table,
                               std::size_t position_offset);

}  // namespace inference_engine
