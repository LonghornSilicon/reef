#include "inference_engine/tensor.hpp"

#include <cstdint>

namespace inference_engine {

template Tensor<float> matmul(const Tensor<float>&, const Tensor<float>&);
template Tensor<std::int8_t> matmul(const Tensor<std::int8_t>&,
                                    const Tensor<std::int8_t>&);

} // namespace inference_engine
