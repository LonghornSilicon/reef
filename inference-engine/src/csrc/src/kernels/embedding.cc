// The embedding operator (operator/embedding.hpp) for Coral NPU: token row
// lookup from a streamed slice of the table, and the position add. See
// kernels.h.
#include "helpers.h"
#include "kernels.h"

extern "C" size_t embedding_lookup(size_t length, size_t width,
                                   const uint32_t* ids, uint32_t vocab_start,
                                   size_t vocab_rows, const int8_t* table,
                                   int8_t* y) {
    size_t filled = 0;
    for (size_t position = 0; position < length; ++position) {
        const uint32_t id = ids[position];
        if (id < vocab_start || id - vocab_start >= vocab_rows) {
            continue; // in another slice of the table
        }
        const int8_t* row =
            table + (static_cast<size_t>(id - vocab_start) * width);
        int8_t* out = y + (position * width);
        size_t vl = 0;
        for (size_t c = 0; c < width; c += vl) {
            vl = rvv::set_vl(width - c);
            rvv::store_i8(out + c, rvv::load_i8(row + c, vl), vl);
        }
        ++filled;
    }
    return filled;
}

extern "C" void add_position_embeddings(size_t length, size_t width,
                                        const int8_t* tokens, float token_scale,
                                        const int8_t* position_table,
                                        float position_scale,
                                        size_t position_offset, float out_scale,
                                        int8_t* y) {
    // token * ft + position * fp with both factors as q-bit fixed point
    const int8_t* positions = position_table + (position_offset * width);
    const float ft = token_scale / out_scale;
    const float fp = position_scale / out_scale;
    const int32_t q = quant::fraction_bits(ft + fp);
    const int32_t ft_q = quant::to_fixed(ft, q);
    const int32_t fp_q = quant::to_fixed(fp, q);
    const size_t n = length * width;
    size_t vl = 0;
    for (size_t i = 0; i < n; i += vl) {
        vl = rvv::set_vl(n - i);
        rvv::vi32 t =
            rvv::mul_x(rvv::widen(rvv::load_i8(tokens + i, vl), vl), ft_q, vl);
        t = rvv::mac_x(t, fp_q, rvv::widen(rvv::load_i8(positions + i, vl), vl),
                       vl);
        rvv::store_i8(y + i, rvv::round_shift_to_int8(t, q, vl), vl);
    }
}
