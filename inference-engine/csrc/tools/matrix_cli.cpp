#include "inference_engine/matrix.hpp"

#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
    std::size_t rows = 0;
    std::size_t inner = 0;
    std::size_t columns = 0;
    if (!(std::cin >> rows >> inner >> columns) || rows == 0 || inner == 0 ||
        columns == 0 || rows > 1024 || inner > 1024 || columns > 1024) {
        std::cerr << "expected three positive dimensions up to 1024\n";
        return 1;
    }

    inference_engine::Tensor left{{rows, inner},
                                  std::vector<float>(rows * inner)};
    inference_engine::Tensor right{{inner, columns},
                                   std::vector<float>(inner * columns)};
    for (auto& value : left.values) {
        if (!(std::cin >> value)) {
            std::cerr << "missing left matrix value\n";
            return 1;
        }
    }
    for (auto& value : right.values) {
        if (!(std::cin >> value)) {
            std::cerr << "missing right matrix value\n";
            return 1;
        }
    }

    try {
        const auto result = inference_engine::matmul(left, right);
        std::cout << result.shape[0] << ' ' << result.shape[1] << '\n';
        std::cout << std::setprecision(
            std::numeric_limits<float>::max_digits10);
        for (const auto value : result.values) {
            std::cout << value << ' ';
        }
        std::cout << '\n';
    } catch (const std::invalid_argument& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
