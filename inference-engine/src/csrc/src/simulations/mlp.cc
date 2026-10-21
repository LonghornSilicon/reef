// The MLP operator (operator/mlp.hpp) for Coral NPU: c_fc, GeLU and c_proj
// over one chunk of the inner width per call, so the weights stream through
// DTCM like attention's K/V blocks. See kernels.h.
#include "helpers.h"
#include "kernels.h"

extern "C" void mlp_block(size_t rows, size_t d, size_t chunk, const int8_t* x,
                          const uint8_t* fc_wt, int32_t fc_zero,
                          const int32_t* fc_bias, float fc_scale,
                          const int8_t* gelu_table, const uint8_t* proj_wt,
                          int32_t proj_zero, int32_t* scratch, int32_t* acc) {
    int32_t* s = scratch; // c_fc, then the c_proj partial sums
    auto* h =
        reinterpret_cast<int8_t*>(scratch + (rows * (chunk > d ? chunk : d)));

    // 1. c_fc for this chunk, requantized to GeLU's input, then GeLU
    linear_acc(rows, d, chunk, x, fc_wt, fc_zero, fc_bias, s);
    for (size_t i = 0; i < rows * chunk; ++i) {
        h[i] = quant::to_int8(fc_scale * static_cast<float>(s[i]), 1.0F);
    }
    gelu(rows * chunk, h, gelu_table, h);

    // 2. this chunk's share of c_proj, added into the running sums
    linear_acc(rows, chunk, d, h, proj_wt, proj_zero, nullptr, s);
    for (size_t i = 0; i < rows * d; ++i) {
        acc[i] += s[i];
    }
}

extern "C" void mlp_finish(size_t rows, size_t d, const int32_t* acc,
                           const int32_t* proj_bias, float acc_scale,
                           float out_scale, int8_t* y) {
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < d; ++c) {
            const int32_t sum = acc[(r * d) + c] + proj_bias[c];
            y[(r * d) + c] =
                quant::to_int8(acc_scale * static_cast<float>(sum), out_scale);
        }
    }
}
