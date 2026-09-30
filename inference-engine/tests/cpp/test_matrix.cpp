#include "inference_engine/matrix.hpp"

#include <stdexcept>

int main() {
    const inference_engine::Tensor left{{2, 3}, {1, 2, 3, 4, 5, 6}};
    const inference_engine::Tensor right{{3, 2}, {7, 8, 9, 10, 11, 12}};
    const auto result = inference_engine::matmul(left, right);
    if (result.shape != std::vector<std::size_t>{2, 2} ||
        result.values != std::vector<float>{58, 64, 139, 154}) {
        return 1;
    }

    try {
        inference_engine::matmul(left, inference_engine::Tensor{{2, 2},
                                                                 {1, 2, 3, 4}});
        return 2;
    } catch (const std::invalid_argument&) {
        // Expected: incompatible inner dimensions.
    }

    try {
        inference_engine::matmul(inference_engine::Tensor{{2, 3}, {1, 2}},
                                 right);
        return 3;
    } catch (const std::invalid_argument&) {
        // Expected: shape does not match storage.
    }

    return 0;
}
