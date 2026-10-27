#pragma once

// Narrowing used by the integer kernels: saturate(rounding_shift(v, shift))
// from int8.hpp. vnclip with vxrm = RNU shifts right, adds the last bit
// shifted out and saturates; going through int16 first saturates the same
// way as going to int8 directly.

#include <riscv_vector.h>

#include <cstddef>

namespace reef_kernels {
namespace detail {

inline vint8m1_t narrow_int8(vint32m4_t v, unsigned shift, std::size_t vl) {
    const vint16m2_t half =
        __riscv_vnclip_wx_i16m2(v, shift, __RISCV_VXRM_RNU, vl);
    return __riscv_vnclip_wx_i8m1(half, 0, __RISCV_VXRM_RNU, vl);
}

} // namespace detail
} // namespace reef_kernels
