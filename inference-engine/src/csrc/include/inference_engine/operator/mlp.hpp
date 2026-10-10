#pragma once

/** @file
 *  @brief Position-wise feed-forward block.
 */

#include "inference_engine/operator/activation.hpp"
#include "inference_engine/operator/linear.hpp"
#include "inference_engine/tensor.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>

namespace inference_engine {

/** Weights and biases for the two projections of the feed-forward block.
 *
 * Names follow the workloads reference (workloads/models/gpt_neo.py):
 * c_fc widens the model width to the inner width, c_proj projects it back.
 * Both weights use the layout linear() expects, [out_features, in_features].
 * An empty bias tensor means that projection has no bias.
 *
 * @tparam Scalar Tensor value type.
 */
template <typename Scalar> struct MlpWeights {
    /// Model width of the input and output rows.
    std::size_t input_size{};
    /// Inner width between the two projections, 4 * input_size in GPT-Neo.
    std::size_t inner_size{};
    /// First projection weights, shaped [inner_size, input_size].
    Tensor<Scalar> c_fc_weight{};
    /// First projection bias, shaped [inner_size], or empty for no bias.
    Tensor<Scalar> c_fc_bias{};
    /// Second projection weights, shaped [input_size, inner_size].
    Tensor<Scalar> c_proj_weight{};
    /// Second projection bias, shaped [input_size], or empty for no bias.
    Tensor<Scalar> c_proj_bias{};
};

namespace detail {

/** Bind a bias tensor for linear(), treating an empty tensor as no bias.
 *
 * @tparam Scalar Tensor value type.
 * @param bias Bias tensor owned by the caller.
 * @return A reference to @p bias, or std::nullopt when it holds no values.
 */
template <typename Scalar>
std::optional<std::reference_wrapper<const Tensor<Scalar>>>
optional_bias(const Tensor<Scalar>& bias) {
    if (bias.values.empty()) {
        return std::nullopt;
    }
    return std::cref(bias);
}

/** Throw unless a tensor is a [rows, columns] matrix with matching storage.
 *
 * @tparam Scalar Tensor value type.
 * @param tensor Matrix to inspect.
 * @param rows Expected first dimension.
 * @param columns Expected second dimension.
 * @param name Parameter name used in the error message.
 * @throws std::invalid_argument If the shape or value count does not match.
 */
template <typename Scalar>
void require_matrix(const Tensor<Scalar>& tensor, std::size_t rows,
                    std::size_t columns, const char* name) {
    if (tensor.shape.size() != 2 || tensor.shape[0] != rows ||
        tensor.shape[1] != columns || !valid_tensor(tensor)) {
        throw std::invalid_argument(
            std::string(name) + " must have shape [" + std::to_string(rows) +
            ", " + std::to_string(columns) + "] with matching storage");
    }
}

} // namespace detail

/** Apply the model's two-projection feed-forward block.
 *
 * @tparam Scalar Tensor value type.
 */
template <typename Scalar> class MLP {
  public:
    /** Bind externally owned layer parameters.
     *
     * @param weights Immutable parameters for both projections.
     */
    explicit MLP(const MlpWeights<Scalar>& weights) : weights_(weights) {}

    /** Run both projections and the activation between them.
     *
     * Computes c_proj(gelu(c_fc(input))), the GPT-Neo feed-forward block.
     *
     * @param input Rows shaped [rows, input_size].
     * @return Rows shaped [rows, input_size].
     * @throws std::invalid_argument If the input or weight shapes do not
     * match the configured widths.
     * @throws std::overflow_error If a product or sum cannot be represented.
     */
    [[nodiscard]] Tensor<Scalar> forward(const Tensor<Scalar>& input) const {
        if (input.shape.size() != 2) {
            throw std::invalid_argument("MLP input must be a 2-D tensor");
        }
        if (input.shape[1] != weights_.input_size) {
            throw std::invalid_argument(
                "MLP input width does not match the weight layout");
        }
        detail::require_matrix(weights_.c_fc_weight, weights_.inner_size,
                               weights_.input_size, "c_fc_weight");
        detail::require_matrix(weights_.c_proj_weight, weights_.input_size,
                               weights_.inner_size, "c_proj_weight");

        const Tensor<Scalar> hidden =
            gelu(linear(input, weights_.c_fc_weight,
                        detail::optional_bias(weights_.c_fc_bias)));
        return linear(hidden, weights_.c_proj_weight,
                      detail::optional_bias(weights_.c_proj_bias));
    }

  private:
    const MlpWeights<Scalar>& weights_;
};

} // namespace inference_engine
