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
    size_t vl = 0;

    // 1. c_fc for this chunk, requantized to GeLU's input, then GeLU
    linear_acc(rows, d, chunk, x, fc_wt, fc_zero, fc_bias, s);
    const quant::Requant to_gelu = quant::requant_params(fc_scale);
    for (size_t i = 0; i < rows * chunk; i += vl) {
        vl = rvv::set_vl((rows * chunk) - i);
        const rvv::vi8 pre =
            rvv::requant(rvv::load_i32(s + i, vl), to_gelu, vl);
        rvv::store_i8(h + i, rvv::lookup_i8(gelu_table, pre, vl), vl);
    }

    // 2. this chunk's share of c_proj, added into the running sums
    linear_acc(rows, chunk, d, h, proj_wt, proj_zero, nullptr, s);
    for (size_t i = 0; i < rows * d; i += vl) {
        vl = rvv::set_vl((rows * d) - i);
        rvv::store_s(
            acc + i,
            rvv::add(rvv::load_i32(acc + i, vl), rvv::load_i32(s + i, vl), vl),
            vl);
    }
}

extern "C" void mlp_finish(size_t rows, size_t d, const int32_t* acc,
                           const int32_t* proj_bias, float acc_scale,
                           float out_scale, int8_t* y) {
    const quant::Requant to_out = quant::requant_params(acc_scale / out_scale);
    size_t vl = 0;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < d; c += vl) {
            vl = rvv::set_vl(d - c);
            const size_t at = (r * d) + c;
            const rvv::vi32 sum =
                rvv::add(rvv::load_i32(acc + at, vl),
                         rvv::load_i32(proj_bias + c, vl), vl);
            rvv::store_i8(y + at, rvv::requant(sum, to_out, vl), vl);
        }
    }
}
