#include "inference_engine/operator/matrix.hpp"

#include <cstdint>

namespace inference_engine {

// Keep common scalar instantiations in the C++ library. The template
// definition stays in the header so other scalar types can use it, too.
template Tensor<float> matmul(const Tensor<float>&, const Tensor<float>&);
template Tensor<std::int8_t> matmul(const Tensor<std::int8_t>&,
                                    const Tensor<std::int8_t>&);

// @ MLP team
// TODO: Implement the linear template in a header using the shared weight
// layout, then test its floating-point and quantized behavior.

} // namespace inference_engine
