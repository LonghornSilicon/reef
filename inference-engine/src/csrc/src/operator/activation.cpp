#include "inference_engine/operator/activation.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace inference_engine{
// @ MLP team
// TODO: Implement gelu declared in activation.hpp.

    // @ Attention team
    //dimension-aware softmax declared in activation.hpp.
    Tensor<float> softmax(const Tensor<float>& input, std::size_t dim){

        const std::vector<std::size_t> shape = input.shape;

        Tensor<float> output;

        //"jumps" calculations -> the tensor data is stored in 1-D row-major order, so we have to calculate "jumps" between data
        const std::size_t axis   = shape[dim];
        const std::size_t stride = std::accumulate(shape.begin() + dim + 1, shape.end(),
                                                std::size_t{1}, std::multiplies<>{});
        const std::size_t outer  = input.values.size() / (axis * stride);


        for (std::size_t o = 0; o < outer; ++o) {
            for (std::size_t i = 0; i < stride; ++i) {
                const std::size_t start = o * axis * stride + i;
                const float* in  = input.values.data() + start;
                float*       out = output.values.data() + start;

                // max (stability shift)
                float m = -std::numeric_limits<float>::infinity();
                for (std::size_t k = 0; k < axis; ++k)
                    m = std::max(m, in[k * stride]);

                if (m == -std::numeric_limits<float>::infinity()) {   // fully masked row
                    for (std::size_t k = 0; k < axis; ++k) out[k * stride] = 0.0f;
                    continue;
                }

                // sum of exp every `stride` steps
                double sum = 0.0;
                for (std::size_t k = 0; k < axis; ++k) {
                    const float e = std::exp(in[k * stride] - m);
                    out[k * stride] = e;
                    sum += e;
                }

                // repeat, dividing by the total
                const float inv = static_cast<float>(1.0 / sum);
                for (std::size_t k = 0; k < axis; ++k)
                    out[k * stride] *= inv;
            }
        }
        return output;
    }
}