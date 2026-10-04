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
        tensor.shape[1] != columns) {
        throw std::invalid_argument(std::string(name) + " must have shape [" +
                                    std::to_string(rows) + ", " +
                                    std::to_string(columns) + "]");
    }
}

void validate_vector_shape(const Tensor<float>& tensor, std::size_t length,
                           const char* name) {
    if (tensor.shape.size() != 1 || tensor.shape[0] != length) {
        throw std::invalid_argument(std::string(name) + " must have shape [" +
                                    std::to_string(length) + "]");
    }
}

} // namespace

MLP::MLP(const MlpWeights& weights) : weights_(weights) {}

Tensor<float> MLP::forward(const Tensor<float>& input) const {
    if (input.shape.size() != 2) {
        throw std::invalid_argument("MLP input must be a 2-D tensor");
    }
    if (input.shape[1] != weights_.input_size) {
        throw std::invalid_argument(
            "MLP input width does not match the weight layout");
    }

    validate_matrix_shape(weights_.fc_weight, weights_.hidden_size,
                          weights_.input_size, "fc_weight");
    if (!weights_.fc_bias.values.empty()) {
        validate_vector_shape(weights_.fc_bias, weights_.hidden_size,
                              "fc_bias");
    }
    validate_matrix_shape(weights_.proj_weight, weights_.input_size,
                          weights_.hidden_size, "proj_weight");
    if (!weights_.proj_bias.values.empty()) {
        validate_vector_shape(weights_.proj_bias, weights_.input_size,
                              "proj_bias");
    }

    const auto rows = input.shape[0];
    const auto columns = input.shape[1];
    Tensor<float> hidden{{rows, weights_.hidden_size},
                         std::vector<float>(rows * weights_.hidden_size, 0.0f)};

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t out = 0; out < weights_.hidden_size; ++out) {
            float sum = 0.0f;
            if (!weights_.fc_bias.values.empty()) {
                sum += weights_.fc_bias.values[out];
            }
            for (std::size_t in = 0; in < columns; ++in) {
                const auto index = (row * columns) + in;
                const auto weight_index = (out * columns) + in;
                sum += input.values[index] *
                       weights_.fc_weight.values[weight_index];
            }
            hidden.values[(row * weights_.hidden_size) + out] = gelu_tanh(sum);
        }
    }

    Tensor<float> output{{rows, columns},
                         std::vector<float>(rows * columns, 0.0f)};

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t out = 0; out < columns; ++out) {
            float sum = 0.0f;
            if (!weights_.proj_bias.values.empty()) {
                sum += weights_.proj_bias.values[out];
            }
            for (std::size_t hidden_index = 0;
                 hidden_index < weights_.hidden_size; ++hidden_index) {
                const auto index = (row * weights_.hidden_size) + hidden_index;
                const auto weight_index =
                    (out * weights_.hidden_size) + hidden_index;
                sum += hidden.values[index] *
                       weights_.proj_weight.values[weight_index];
            }
            output.values[(row * columns) + out] = sum;
        }
    }

    return output;
}

} // namespace inference_engine
