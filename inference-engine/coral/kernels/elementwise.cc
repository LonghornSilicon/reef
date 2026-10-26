// Embedding lookup, residual add and argmax, in float and int8.

#include "reef_kernels/kernels.hpp"

#include <riscv_vector.h>

namespace reef_kernels {

void embedding_f32(std::size_t width, std::uint32_t id, std::uint32_t position,
                   const float* wte, const float* wpe, float* y) {
    add_f32(width, wte + (static_cast<std::size_t>(id) * width),
            wpe + (static_cast<std::size_t>(position) * width), y);
}

void add_f32(std::size_t n, const float* a, const float* b, float* y) {
    for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m4(n - i);
        __riscv_vse32_v_f32m4(
            y + i,
            __riscv_vfadd_vv_f32m4(__riscv_vle32_v_f32m4(a + i, vl),
                                   __riscv_vle32_v_f32m4(b + i, vl), vl),
            vl);
        i += vl;
    }
}

std::uint32_t argmax_f32(std::size_t n, const float* x) {
    vfloat32m1_t largest = __riscv_vfmv_s_f_f32m1(x[0], 1);
    for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m4(n - i);
        largest = __riscv_vfredmax_vs_f32m4_f32m1(
            __riscv_vle32_v_f32m4(x + i, vl), largest, vl);
        i += vl;
    }
    const float best = __riscv_vfmv_f_s_f32m1_f32(largest);
    for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m4(n - i);
        const vbool8_t hit = __riscv_vmfeq_vf_f32m4_b8(
            __riscv_vle32_v_f32m4(x + i, vl), best, vl);
        const long lane = __riscv_vfirst_m_b8(hit, vl);
        if (lane >= 0) {
            return static_cast<std::uint32_t>(i +
                                              static_cast<std::size_t>(lane));
        }
        i += vl;
    }
    return 0;
}

void add_int8(std::size_t n, const std::int8_t* a, const std::int8_t* b,
              std::int8_t* y) {
    for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e8m1(n - i);
        __riscv_vse8_v_i8m1(
            y + i,
            __riscv_vsadd_vv_i8m1(__riscv_vle8_v_i8m1(a + i, vl),
                                  __riscv_vle8_v_i8m1(b + i, vl), vl),
            vl);
        i += vl;
    }
}

void embedding_int8(std::size_t width, std::uint32_t id, std::uint32_t position,
                    const std::int8_t* wte, const std::int8_t* wpe,
                    std::int8_t* y) {
    add_int8(width, wte + (static_cast<std::size_t>(id) * width),
             wpe + (static_cast<std::size_t>(position) * width), y);
}

} // namespace reef_kernels
