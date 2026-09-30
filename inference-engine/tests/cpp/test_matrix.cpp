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

    bool rejected_incompatible_dimensions = false;
    try {
        inference_engine::matmul(
            left, inference_engine::Tensor{{2, 2}, {1, 2, 3, 4}});
    } catch (const std::invalid_argument&) {
        rejected_incompatible_dimensions = true;
    }
    if (!rejected_incompatible_dimensions)
        return 2;

    bool rejected_bad_storage = false;
    try {
        inference_engine::matmul(inference_engine::Tensor{{2, 3}, {1, 2}},
                                 right);
    } catch (const std::invalid_argument&) {
        rejected_bad_storage = true;
    }
    if (!rejected_bad_storage)
        return 3;

    return 0;
}
