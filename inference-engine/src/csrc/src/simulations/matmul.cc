#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

// [n, inner] * [inner, m] -> [n, m]
extern "C" void matmul(size_t n, size_t inner, size_t m, const uint8_t* a,
                       const uint8_t* b, uint32_t* c) {
    // assert(m == __riscv_vsetvlmax_e8m1(m));
    size_t vl = __riscv_vsetvl_e8m1(
        m); // our vectors will be uint8s stored in 1 register group

    // each row of C can be calculated at once
    for (size_t i = 0; i < n; i++) {
        vuint32m4_t acc = __riscv_vmv_v_x_u32m4(0, vl);
        for (size_t j = 0; j < inner; j++) {
            vuint8m1_t b8 = __riscv_vle8_v_u8m1(b + (j * m), vl);
            vuint16m2_t b16 = __riscv_vzext_vf2_u16m2(b8, vl);
            acc =
                __riscv_vwmaccu_vx_u32m4(acc, *(a + (i * inner) + j), b16, vl);
        }
        __riscv_vse32_v_u32m4(c + (i * m), acc, vl);
    }
}
