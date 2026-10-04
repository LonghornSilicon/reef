#include "inference_engine/operator/mlp.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace inference_engine {
namespace {

float gelu_tanh(float x) {
    constexpr float kCoefficient = 0.7978845608028654f;
    constexpr float kCubic = 0.044715f;
    return 0.5f * x *
           (1.0f + std::tanh(kCoefficient * (x + kCubic * x * x * x)));
}

void validate_matrix_shape(const Tensor<float>& tensor, std::size_t rows,
                           std::size_t columns, const char* name) {
    if (tensor.shape.size() != 2 || tensor.shape[0] != rows ||
        tensor.shape[1] != columns || tensor.values.size() != rows * columns) {
        throw std::invalid_argument(
            std::string(name) + " must have shape [" + std::to_string(rows) +
            ", " + std::to_string(columns) + "] and " + "matching storage");
    }
}

void validate_vector_shape(const Tensor<float>& tensor, std::size_t length,
                           const char* name) {
    if (tensor.shape.size() != 1 || tensor.shape[0] != length ||
        tensor.values.size() != length) {
        throw std::invalid_argument(std::string(name) + " must have shape [" +
                                    std::to_string(length) + "] and " +
                                    "matching storage");
    }
}

} // namespace

MLP::MLP(const MlpWeights& weights) : weights_(weights) {}

Tensor<float> MLP::forward(const Tensor<float>& input) const {
    // Standard GPT-Neo feed-forward block:
    //   W_2 GELU(W_1 x + b_1) + b_2
    if (input.shape.size() != 2) {
        throw std::invalid_argument("MLP input must be a 2-D tensor");
    }
    if (input.shape[1] != weights_.input_size) {
        throw std::invalid_argument(
            "MLP input width does not match the weight layout");
    }

    validate_matrix_shape(weights_.W_1, weights_.hidden_size,
                          weights_.input_size, "W_1");
    if (!weights_.b_1.values.empty()) {
        validate_vector_shape(weights_.b_1, weights_.hidden_size, "b_1");
    }
    validate_matrix_shape(weights_.W_2, weights_.input_size,
                          weights_.hidden_size, "W_2");
    if (!weights_.b_2.values.empty()) {
        validate_vector_shape(weights_.b_2, weights_.input_size, "b_2");
    }

    const size_t rows = input.shape[0];
    const size_t columns = input.shape[1];
    Tensor<float> hidden{{rows, weights_.hidden_size},
                         std::vector<float>(rows * weights_.hidden_size, 0.0f)};

    // layer 1: r_1 = GELU(W_1 * x + b_1)
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t out = 0; out < weights_.hidden_size; ++out) {
            float sum = 0.0f;
            if (!weights_.b_1.values.empty()) {
                sum += weights_.b_1.values[out];
            }
            for (std::size_t in = 0; in < columns; ++in) {
                const std::size_t index = (row * columns) + in;
                const std::size_t weight_index = (out * columns) + in;
                sum += input.values[index] * weights_.W_1.values[weight_index];
            }
            hidden.values[(row * weights_.hidden_size) + out] = gelu_tanh(sum);
        }
    }

    Tensor<float> output{{rows, columns},
                         std::vector<float>(rows * columns, 0.0f)};

    // matmul 2: r_2 = W_2*r_1 + b_2
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t out = 0; out < columns; ++out) {
            float sum = 0.0f;
            if (!weights_.b_2.values.empty()) {
                sum += weights_.b_2.values[out];
            }
            for (std::size_t hidden_index = 0;
                 hidden_index < weights_.hidden_size; ++hidden_index) {
                const std::size_t index =
                    (row * weights_.hidden_size) + hidden_index;
                const std::size_t weight_index =
                    (out * weights_.hidden_size) + hidden_index;
                sum += hidden.values[index] * weights_.W_2.values[weight_index];
            }
            output.values[(row * columns) + out] = sum;
        }
    }

    return output;
}

} // namespace inference_engine
