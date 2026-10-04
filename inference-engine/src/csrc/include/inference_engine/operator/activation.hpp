#pragma once

/** @file
 *  @brief Elementwise activations and dimension-aware softmax.
 */

#include "tensor.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>

namespace inference_engine {

/** Apply the tanh-approximation GELU elementwise.
 *
 * @param input Input values.
 * @return Tensor with the same shape after activation.
 */
// @ MLP team
// TODO: Implement and set a numeric tolerance for host and Coral comparisons.
template <typename Scalar> Tensor<Scalar> gelu(const Tensor<Scalar>& input);

/** Apply softmax along one tensor dimension, including axis zero for 1-D data.
 *
 * @tparam Scalar Tensor value type.
 * @param input Input values.
 * @param dim Zero-based dimension to normalize.
 * @return Tensor with the same shape and normalized values along @p dim.
 */
// @ Attention team
// TODO: Implement a numerically stable operation; define masked-row behavior
// and accumulation precision. Define a scale for integer output tensors.
template <typename Scalar>
Tensor<Scalar> softmax(const Tensor<Scalar>& input, std::size_t dim) {

    const std::vector<std::size_t> shape = input.shape;

    Tensor<Scalar> output;

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
            Scalar m = -std::numeric_limits<Scalar>::infinity();
            for (std::size_t k = 0; k < axis; ++k) {
                m = std::max(m, in[k * stride]);
            }

            // skip fully masked row
            if (m == -std::numeric_limits<Scalar>::infinity()) {
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
