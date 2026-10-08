#pragma once

/** @file
 *  @brief Dense affine projection shared by the attention and MLP blocks.
 */

#include "inference_engine/tensor.hpp"
#include "inference_engine/util.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>

namespace inference_engine {

/** Apply a weight matrix and an optional bias to input rows.
 *
 * Computes input @ weight.T + bias. The transpose is the weight layout the
 * workloads reference uses (see workloads/operators/linear.py, matching
 * torch.nn.Linear): @p weight holds one row per output feature, so a
 * [out_features, in_features] matrix maps [rows, in_features] input to
 * [rows, out_features] output. Passing a weight in the opposite layout is
 * rejected by matmul rather than silently transposing the projection.
 *
 * @tparam Scalar Tensor value type.
 * @param input Input rows shaped [rows, in_features].
 * @param weight Output weights shaped [out_features, in_features].
 * @param bias Optional reference to a bias shaped [out_features]; no copy is
 *        made.
 * @return Projected rows shaped [rows, out_features].
 * @throws std::invalid_argument If the shapes or value counts are
 * incompatible, or the bias is not one value per output feature.
 * @throws std::overflow_error If a product or sum cannot be represented.
 */
template <typename Scalar>
Tensor<Scalar>
linear(const Tensor<Scalar>& input, const Tensor<Scalar>& weight,
       std::optional<std::reference_wrapper<const Tensor<Scalar>>> bias =
           std::nullopt) {
    if (weight.shape.size() != 2) {
        throw std::invalid_argument("linear weight must be a 2-D matrix");
    }
    Tensor<Scalar> output = matmul(input, transpose(weight, 0, 1));
    if (!bias.has_value()) {
        return output;
    }

    const Tensor<Scalar>& values = bias->get();
    const std::size_t out_features = output.shape[1];
    if (values.shape.size() != 1 || values.shape[0] != out_features ||
        !valid_tensor(values)) {
        throw std::invalid_argument(
            "linear bias must hold one value per output feature");
    }
    for (std::size_t row = 0; row < output.shape[0]; ++row) {
        for (std::size_t column = 0; column < out_features; ++column) {
            Scalar& value = output.values[(row * out_features) + column];
            value = safe_add(value, values.values[column]);
        }
    }
    return output;
}

} // namespace inference_engine
