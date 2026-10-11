// RVV embedding kernels for Coral NPU, built on rvv_helpers.h
//
// Both kernels are pure mem ops: one row copy and one row add per
// token, no MACs. They walk each row in strips of vl bytes
// (vl <= 16 with VLEN = 128)
#include "embedding_rvv.h"
#include "rvv_helpers.h"

extern "C" void embedding_lookup_rvv(size_t length, size_t width,
                                     const uint32_t* ids, const int8_t* table,
                                     int8_t* out) {
    for (size_t i = 0; i < length; i++) {
        const int8_t* src = table + (ids[i] * width);
        int8_t* dst = out + (i * width);
        size_t vl = 0;
        for (size_t c = 0; c < width; c += vl) {
            vl = rvv::set_vl(width - c);
            rvv::store_s8(dst + c, rvv::load_s8(src + c, vl), vl);
        }
    }
}

extern "C" void embedding_add_position_rvv(size_t length, size_t width,
                                           size_t position_offset,
                                           const int8_t* position_table,
                                           int8_t* out) {
    for (size_t i = 0; i < length; i++) {
        const int8_t* src = position_table + ((position_offset + i) * width);
        int8_t* dst = out + (i * width);
        size_t vl = 0;
        for (size_t c = 0; c < width; c += vl) {
            vl = rvv::set_vl(width - c);
            const rvv::vs8 sum = rvv::add_s8(rvv::load_s8(dst + c, vl),
                                             rvv::load_s8(src + c, vl), vl);
            rvv::store_s8(dst + c, sum, vl);
        }
    }
}