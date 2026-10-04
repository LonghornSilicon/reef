#include "inference_engine/tensor.hpp"

#include <stdexcept>

namespace inference_engine {

namespace {

std::size_t element_count(const std::vector<std::size_t>& shape) {
    std::size_t count = 1;
    for (const std::size_t dim : shape) {
        count *= dim;
    }
    return count;
}

std::vector<std::size_t>
row_major_strides(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t axis = shape.size(); axis > 1; --axis) {
        strides[axis - 2] = strides[axis - 1] * shape[axis - 1];
    }
    return strides;
}

} // namespace

Tensor<float> transpose(const Tensor<float>& tensor, std::size_t dim0,
                        std::size_t dim1) {
    const std::size_t rank = tensor.shape.size();
    if (dim0 >= rank || dim1 >= rank) {
        throw std::invalid_argument(
            "transpose dims must be less than the rank");
    }
    if (element_count(tensor.shape) != tensor.values.size()) {
        throw std::invalid_argument("transpose shape does not match values");
    }

    std::vector<std::size_t> out_shape = tensor.shape;
    out_shape[dim0] = tensor.shape[dim1];
    out_shape[dim1] = tensor.shape[dim0];

    const std::vector<std::size_t> in_strides = row_major_strides(tensor.shape);
    const std::vector<std::size_t> out_strides = row_major_strides(out_shape);

    Tensor<float> result{out_shape, std::vector<float>(tensor.values.size())};
    for (std::size_t in = 0; in < tensor.values.size(); ++in) {
        std::size_t out = 0;
        for (std::size_t axis = 0; axis < rank; ++axis) {
            const std::size_t coord =
                (in / in_strides[axis]) % tensor.shape[axis];
            std::size_t out_axis = axis;
            if (axis == dim0) {
                out_axis = dim1;
            } else if (axis == dim1) {
                out_axis = dim0;
            }
            out += coord * out_strides[out_axis];
        }
        result.values[out] = tensor.values[in];
    }
    return result;
}

Tensor<float> reshape(const Tensor<float>& tensor,
                      const std::vector<std::size_t>& shape) {
    if (element_count(shape) != tensor.values.size()) {
        throw std::invalid_argument("reshape shape does not match values");
    }
    return Tensor<float>{shape, tensor.values};
}

} // namespace inference_engine
