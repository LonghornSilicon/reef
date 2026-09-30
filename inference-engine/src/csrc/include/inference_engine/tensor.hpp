#pragma once

#include <cstddef>
#include <vector>

namespace inference_engine {

// Small host-side tensor for the first matrix-multiply slice. Values are
// row-major for 2-D tensors. Other shapes are not supported by matmul yet.
// TODO: Agree across teams on the long-term shape, scalar type, and ownership.
struct Tensor {
    std::vector<std::size_t> shape;
    std::vector<float> values;
};

} // namespace inference_engine
