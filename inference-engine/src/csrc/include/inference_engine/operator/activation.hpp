#pragma once

/** @file
 *  @brief Elementwise activations and dimension-aware softmax.
 */

#include "inference_engine/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace inference_engine {

/** Apply the tanh-approximation GELU elementwise.
 *
 * Uses the gelu_pytorch_tanh form the workloads reference pins (see
 * workloads/operators/activation.py), not the erf form:
 *   0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
 *
 * @tparam Scalar Floating-point tensor value type. Integer tensors need an
 * agreed output scale first, so they fail to compile here rather than
 * silently truncating.
 * @param input Input values.
 * @return Tensor with the same shape after activation.
 * @throws std::invalid_argument If the shape does not match the value count.
 */
template <typename Scalar> Tensor<Scalar> gelu(const Tensor<Scalar>& input) {
    static_assert(std::is_floating_point_v<Scalar>,
                  "gelu requires a floating-point scalar; integer tensors "
                  "need an agreed output scale first");
    if (detail::element_count(input.shape) != input.values.size()) {
        throw std::invalid_argument("gelu shape does not match values");
    }

    /// sqrt(2 / pi).
    constexpr auto kCoefficient = static_cast<Scalar>(0.7978845608028654);
    constexpr auto kCubic = static_cast<Scalar>(0.044715);
    constexpr auto kHalf = static_cast<Scalar>(0.5);
    constexpr auto kOne = static_cast<Scalar>(1);

    Tensor<Scalar> output{input.shape,
                          std::vector<Scalar>(input.values.size())};
    for (std::size_t index = 0; index < input.values.size(); ++index) {
        const Scalar value = input.values[index];
        const Scalar inner =
            kCoefficient * (value + (kCubic * value * value * value));
        output.values[index] = kHalf * value * (kOne + std::tanh(inner));
    }
    return output;
}

/** Apply softmax along one tensor dimension, including axis zero for 1-D data.
 *
 * Every normalized slice must contain at least one finite value: a slice of
 * only -infinity values has no defined distribution and yields NaN.
 *
 * @tparam Scalar Tensor value type.
 * @param input Input values.
 * @param dim Zero-based dimension to normalize.
 * @return Tensor with the same shape and normalized values along @p dim.
 */
// @ Attention team
// TODO: Define accumulation precision and a scale for integer output tensors.
template <typename Scalar>
Tensor<Scalar> softmax(const Tensor<Scalar>& input, std::size_t dim) {

    const std::vector<std::size_t> shape = input.shape;

    Tensor<Scalar> output{shape, std::vector<Scalar>(input.values.size())};

    //"jumps" calculations bc the tensor data is stored in 1-D row-major order
    const std::size_t axis = shape.at(dim);
    const std::size_t stride =
        std::accumulate(shape.begin() + static_cast<std::ptrdiff_t>(dim) + 1,
                        shape.end(), std::size_t{1}, std::multiplies<>{});
    const std::size_t outer = input.values.size() / (axis * stride);

    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t i = 0; i < stride; ++i) {
            // the k-th element along the normalized dimension lives at
            // start + (k * stride)
            const std::size_t start = (o * axis * stride) + i;

            // find max for numerical stability
            Scalar m = input.values.at(start);
            for (std::size_t k = 1; k < axis; ++k) {
                m = std::max(m, input.values.at(start + (k * stride)));
            }

            // sum of exp
            Scalar sum = 0;
            for (std::size_t k = 0; k < axis; ++k) {
                const std::size_t index = start + (k * stride);
                const Scalar e = std::exp(input.values.at(index) - m);
                output.values.at(index) = e;
                sum += e;
            }

            // normalize
            for (std::size_t k = 0; k < axis; ++k) {
                output.values.at(start + (k * stride)) /= sum;
            }
        }
    }
    return output;
}

/** Softmax over the last dimension, restricted to visible positions.
 *
 * Masked positions receive weight zero and take no part in the maximum or
 * the normalizing sum, so no sentinel score values are involved. The mask
 * covers the trailing two dimensions and repeats across any leading ones.
 *
 * @tparam Scalar Floating-point tensor value type. Integer tensors need an
 * agreed output scale first, so they fail to compile here rather than
 * silently truncating.
 * @param scores [..., rows, keys] scores to normalize per row.
 * @param visible [rows, keys] mask, true where a position participates.
 * @return Tensor shaped like @p scores; each row's visible weights sum to 1.
 * @throws std::invalid_argument If the mask does not match the trailing
 * score dimensions or a row has no visible position.
 */
template <typename Scalar>
Tensor<Scalar> masked_softmax(const Tensor<Scalar>& scores,
                              const Tensor<bool>& visible) {
    static_assert(std::is_floating_point_v<Scalar>,
                  "masked_softmax requires a floating-point scalar; integer "
                  "tensors need an agreed output scale first");
    const std::size_t rank = scores.shape.size();
    if (rank < 2 || visible.shape.size() != 2 ||
        visible.shape[0] != scores.shape[rank - 2] ||
        visible.shape[1] != scores.shape[rank - 1] ||
        detail::element_count(scores.shape) != scores.values.size() ||
        detail::element_count(visible.shape) != visible.values.size()) {
        throw std::invalid_argument(
            "masked_softmax mask must match the trailing score dimensions");
    }

    const std::size_t num_queries = visible.shape[0];
    const std::size_t num_keys = visible.shape[1];
    const std::size_t num_rows = scores.values.size() / num_keys;

    Tensor<Scalar> output{scores.shape,
                          std::vector<Scalar>(scores.values.size())};
    for (std::size_t row = 0; row < num_rows; ++row) {
        // the k-th key of this row lives at start + k; its visibility at
        // mask_start + k, repeating the mask across leading dimensions
        const std::size_t start = row * num_keys;
        const std::size_t mask_start = (row % num_queries) * num_keys;

        // find max for numerical stability, over the visible positions only
        bool any_visible = false;
        Scalar m{};
        for (std::size_t k = 0; k < num_keys; ++k) {
            if (visible.values.at(mask_start + k)) {
                m = any_visible ? std::max(m, scores.values.at(start + k))
                                : scores.values.at(start + k);
                any_visible = true;
            }
        }
        if (!any_visible) {
            throw std::invalid_argument(
                "masked_softmax row has no visible position");
        }

        // sum of exp over the visible positions; masked ones weigh zero
        Scalar sum = 0;
        for (std::size_t k = 0; k < num_keys; ++k) {
            const std::size_t index = start + k;
            if (visible.values.at(mask_start + k)) {
                const Scalar e = std::exp(scores.values.at(index) - m);
                output.values.at(index) = e;
                sum += e;
            } else {
                output.values.at(index) = 0;
            }
        }

        // normalize the visible positions
        for (std::size_t k = 0; k < num_keys; ++k) {
            if (visible.values.at(mask_start + k)) {
                output.values.at(start + k) /= sum;
            }
        }
    }
    return output;
}

} // namespace inference_engine
