#pragma once

/** @file
 *  @brief Elementwise activations and dimension-aware softmax.
 */

#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace inference_engine {

/** Apply the tanh-approximation GELU elementwise.
 *
 * @param input Input values.
 * @return Tensor with the same shape after activation.
 */
// @ MLP team
// TODO: Implement and set a numeric tolerance for host and Coral comparisons.
template <typename Scalar> Tensor<Scalar> gelu(const Tensor<Scalar>& input);

/** Score written to disallowed attention positions and treated by softmax as
 * fully masked: negative infinity when the scalar type has one, otherwise the
 * lowest finite value (integral types have no infinity, and negating their
 * zero "infinity" would make masked positions legal scores).
 *
 * @tparam Scalar Tensor value type.
 * @return The masking sentinel for @p Scalar.
 */
template <typename Scalar> constexpr Scalar masked_score() {
    return std::numeric_limits<Scalar>::has_infinity
               ? -std::numeric_limits<Scalar>::infinity()
               : std::numeric_limits<Scalar>::lowest();
}

/** Apply softmax along one tensor dimension, including axis zero for 1-D data.
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
            const std::size_t start = (o * axis * stride) + i;
            const Scalar* in = input.values.data() + start;
            Scalar* out = output.values.data() + start;

            // find max
            Scalar m = masked_score<Scalar>();
            for (std::size_t k = 0; k < axis; ++k) {
                m = std::max(m, in[k * stride]);
            }

            // skip fully masked row
            if (m == masked_score<Scalar>()) {
                for (std::size_t k = 0; k < axis; ++k) {
                    out[k * stride] = 0;
                }
                continue;
            }

            // sum of exp every `stride` steps
            Scalar sum = 0;
            for (std::size_t k = 0; k < axis; ++k) {
                const Scalar e = std::exp(in[k * stride] - m);
                out[k * stride] = e;
                sum += e;
            }

            // repeat, dividing by the total
            for (std::size_t k = 0; k < axis; ++k) {
                out[k * stride] /= sum;
            }
        }
    }
    return output;
}

} // namespace inference_engine
