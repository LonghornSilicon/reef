#include "inference_engine/operator/matrix.hpp"

#include <limits>
#include <stdexcept>

namespace inference_engine {

// @ MLP team
Tensor matmul(const Tensor& left, const Tensor& right) {
    const auto valid_matrix = [](const Tensor& tensor) {
        return tensor.shape.size() == 2 && tensor.shape[0] > 0 &&
               tensor.shape[1] > 0 &&
               tensor.shape[0] <=
                   std::numeric_limits<std::size_t>::max() / tensor.shape[1] &&
               tensor.values.size() == tensor.shape[0] * tensor.shape[1];
    };
    if (!valid_matrix(left) || !valid_matrix(right) ||
        left.shape[1] != right.shape[0] ||
        left.shape[0] >
            std::numeric_limits<std::size_t>::max() / right.shape[1]) {
        throw std::invalid_argument("matmul requires compatible 2-D matrices");
    }

    const auto rows = left.shape[0];
    const auto inner = left.shape[1];
    const auto columns = right.shape[1];
    Tensor result{{rows, columns}, std::vector<float>(rows * columns, 0.0F)};
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            for (std::size_t index = 0; index < inner; ++index) {
                result.values[row * columns + column] +=
                    left.values[row * inner + index] *
                    right.values[index * columns + column];
            }
        }
    }
    return result;
}

// @ MLP team
// TODO: Implement linear declared in matrix.hpp using the shared weight layout.

} // namespace inference_engine
