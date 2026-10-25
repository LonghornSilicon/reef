#pragma once

/** @file
 *  @brief Layer normalization over the last dimension.
 */

#include "inference_engine/tensor.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace inference_engine {

/** Normalize each row over the last dimension, then apply a gain and shift.
 *
 * Matches the workloads reference (workloads/operators/normalization.py):
 *   (x - mean) / sqrt(variance + eps) * weight + bias
 * where variance is the mean squared deviation (biased, like torch).
 *
 * @tparam Scalar Floating-point tensor value type.
 * @param input Input shaped [..., width].
 * @param weight Gain shaped [width].
 * @param bias Optional reference to a shift shaped [width]; no copy is made.
 * @param eps Added to the variance before the square root.
 * @return Tensor with the same shape as @p input.
 * @throws std::invalid_argument If the input is invalid or the gain or shift
 * is not one value per element of the last dimension.
 */
template <typename Scalar>
Tensor<Scalar>
layernorm(const Tensor<Scalar>& input, const Tensor<Scalar>& weight,
          std::optional<std::reference_wrapper<const Tensor<Scalar>>> bias =
              std::nullopt,
          Scalar eps = static_cast<Scalar>(1e-5)) {
    static_assert(std::is_floating_point_v<Scalar>,
                  "layernorm requires a floating-point scalar");
    if (input.shape.empty() || !valid_tensor(input)) {
        throw std::invalid_argument("layernorm input is not a valid tensor");
    }
    const std::size_t width = input.shape.back();
    const auto per_feature = [width](const Tensor<Scalar>& tensor) {
        return tensor.shape.size() == 1 && tensor.shape[0] == width &&
               valid_tensor(tensor);
    };
    if (!per_feature(weight) || (bias.has_value() && !per_feature(*bias))) {
        throw std::invalid_argument(
            "layernorm weight and bias must hold one value per feature");
    }

    Tensor<Scalar> output{input.shape, input.values};
    for (std::size_t start = 0; start < output.values.size(); start += width) {
        Scalar mean = 0;
        for (std::size_t k = 0; k < width; ++k) {
            mean += output.values[start + k];
        }
        mean /= static_cast<Scalar>(width);

        Scalar variance = 0;
        for (std::size_t k = 0; k < width; ++k) {
            const Scalar centred = output.values[start + k] - mean;
            variance += centred * centred;
        }
        variance /= static_cast<Scalar>(width);

        const Scalar inv_std = 1 / std::sqrt(variance + eps);
        for (std::size_t k = 0; k < width; ++k) {
            Scalar& value = output.values[start + k];
            value = (value - mean) * inv_std * weight.values[k];
            if (bias.has_value()) {
                value += bias->get().values[k];
            }
        }
    }
    return output;
}

} // namespace inference_engine
