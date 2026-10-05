#include "inference_engine/operator/embedding.hpp"

#include <cstddef>

namespace inference_engine {

template Tensor<float> lookup_token_embeddings(const TokenIds&,
                                               const Tensor<float>&);
template Tensor<float> add_position_embeddings(const Tensor<float>&,
                                               const Tensor<float>&,
                                               std::size_t);

} // namespace inference_engine
