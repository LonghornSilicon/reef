// Linear (operator/linear.hpp) on Coral M3, ready for reef_perf:
// int8 y = x . W^T + bias, accumulated in int32 and rounded down by 2^SHIFT.
#include <riscv_vector.h>
#include <stdint.h>

#define ROWS 16
#define IN 64
#define OUT 64
#define SHIFT 8

int8_t x[ROWS * IN];
int8_t wt[IN * OUT];  // W^T, so each row of wt is contiguous over outputs
int32_t bias[OUT];
int8_t y[ROWS * OUT];

// int32 -> int8: round half up after >> shift, then saturate
vint8m1_t narrow(vint32m4_t v, int shift, size_t vl) {
    vint16m2_t h = __riscv_vnclip_wx_i16m2(v, shift, __RISCV_VXRM_RNU, vl);
    return __riscv_vnclip_wx_i8m1(h, 0, __RISCV_VXRM_RNU, vl);
}

// 16 outputs per strip: acc[c] += x[r][k] * wt[k][c] over k
void linear(int rows, int in, int out, const int8_t* x, const int8_t* wt,
            const int32_t* bias, int shift, int8_t* y) {
    for (int c = 0; c < out;) {
        size_t vl = __riscv_vsetvl_e8m1(out - c);
        for (int r = 0; r < rows; ++r) {
            vint32m4_t acc = __riscv_vle32_v_i32m4(bias + c, vl);
            for (int k = 0; k < in; ++k) {
                vint16m2_t w = __riscv_vsext_vf2_i16m2(
                    __riscv_vle8_v_i8m1(wt + k * out + c, vl), vl);
                acc = __riscv_vwmacc_vx_i32m4(acc, x[r * in + k], w, vl);
            }
            __riscv_vse8_v_i8m1(y + r * out + c, narrow(acc, shift, vl), vl);
        }
        c += vl;
    }
}

uint32_t seed = 1;
int8_t rnd() {
    seed = seed * 1664525u + 1013904223u;
    return (int8_t)(seed >> 24);
}

int main() {
    for (int i = 0; i < ROWS * IN; ++i) x[i] = rnd();
    for (int i = 0; i < IN * OUT; ++i) wt[i] = rnd();
    for (int i = 0; i < OUT; ++i) bias[i] = rnd() * 16;
    linear(ROWS, IN, OUT, x, wt, bias, SHIFT, y);
#ifdef CHECK  // scalar reference; a mismatch traps reef_perf
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < OUT; ++c) {
            int32_t acc = bias[c];
            for (int k = 0; k < IN; ++k) acc += x[r * IN + k] * wt[k * OUT + c];
            acc = (acc >> SHIFT) + ((acc >> (SHIFT - 1)) & 1);
            acc = acc > 127 ? 127 : acc < -128 ? -128 : acc;
            if (y[r * OUT + c] != acc) asm volatile("ebreak");
        }
    }
#endif
    return 0;
}

// No C runtime: set the stack, turn on the FPU and vector unit, run main,
// then mpause (stops reef_perf).
asm(R"(
    .section .text.start, "ax"
    .globl _start
_start:
    la    sp, _stack_top
    li    t0, (1 << 13) | (1 << 9)
    csrs  mstatus, t0
    call  main
    .word 0x08000073
)");
