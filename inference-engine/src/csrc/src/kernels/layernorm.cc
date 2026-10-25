// LayerNorm (operator/layernorm.hpp) on Coral M3, ready for reef_perf, all
// integer. Per row of WIDTH int8 values:
//   c = x - mean,  std = isqrt(mean(c^2))
//   y = 16 * c / std * gain / 32 + beta    int8 (normalized value at scale 1/16)
// gain is Q5 (32 = 1.0); beta is at the output scale.
#include <riscv_vector.h>
#include <stdint.h>

#define ROWS 16
#define WIDTH 64  // a power of two, so the means are shifts
#define LOG2_WIDTH 6

int8_t x[ROWS * WIDTH];
int8_t gain[WIDTH];
int8_t beta[WIDTH];
int8_t y[ROWS * WIDTH];

vint32m4_t load_i32(const int8_t* p, size_t vl) {
    return __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(p, vl), vl);
}

// floor(sqrt(v)), one result bit per step
uint32_t isqrt(uint32_t v) {
    uint32_t r = 0;
    for (uint32_t bit = 1u << 30; bit != 0; bit >>= 2) {
        if (v >= r + bit) {
            v -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
    }
    return r;
}

void layernorm(int rows, const int8_t* x, const int8_t* gain, const int8_t* beta, int8_t* y) {
    size_t vl;
    for (int r = 0; r < rows; ++r) {
        const int8_t* xr = x + r * WIDTH;
        int32_t sum = 0;
        for (int c = 0; c < WIDTH; c += vl) {
            vl = __riscv_vsetvl_e8m1(WIDTH - c);
            sum = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(
                load_i32(xr + c, vl), __riscv_vmv_s_x_i32m1(sum, vl), vl));
        }
        const int32_t mean = sum >> LOG2_WIDTH;

        int32_t sq = 0;
        for (int c = 0; c < WIDTH; c += vl) {
            vl = __riscv_vsetvl_e8m1(WIDTH - c);
            vint32m4_t d = __riscv_vsub_vx_i32m4(load_i32(xr + c, vl), mean, vl);
            sq = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(
                __riscv_vmul_vv_i32m4(d, d, vl), __riscv_vmv_s_x_i32m1(sq, vl), vl));
        }
        uint32_t sd = isqrt((uint32_t)sq >> LOG2_WIDTH);
        const int32_t inv = (16 << 16) / (int32_t)(sd != 0 ? sd : 1);  // 16 / std in Q16

        for (int c = 0; c < WIDTH; c += vl) {
            vl = __riscv_vsetvl_e8m1(WIDTH - c);
            vint32m4_t d = __riscv_vsub_vx_i32m4(load_i32(xr + c, vl), mean, vl);
            vint32m4_t n = __riscv_vsra_vx_i32m4(__riscv_vmul_vx_i32m4(d, inv, vl), 16, vl);
            vint32m4_t v = __riscv_vmul_vv_i32m4(n, load_i32(gain + c, vl), vl);
            v = __riscv_vadd_vv_i32m4(v, __riscv_vsll_vx_i32m4(load_i32(beta + c, vl), 5, vl), vl);
            vint16m2_t h = __riscv_vnclip_wx_i16m2(v, 5, __RISCV_VXRM_RNU, vl);
            __riscv_vse8_v_i8m1(y + r * WIDTH + c, __riscv_vnclip_wx_i8m1(h, 0, __RISCV_VXRM_RNU, vl), vl);
        }
    }
}

uint32_t seed = 1;
int8_t rnd() {
    seed = seed * 1664525u + 1013904223u;
    return (int8_t)(seed >> 24);
}

int main() {
    for (int i = 0; i < ROWS * WIDTH; ++i) x[i] = rnd();
    for (int i = 0; i < WIDTH; ++i) gain[i] = 32 + rnd() / 8;  // ~1.0
    for (int i = 0; i < WIDTH; ++i) beta[i] = rnd() / 8;
    layernorm(ROWS, x, gain, beta, y);
#ifdef CHECK  // scalar reference; a mismatch traps reef_perf
    for (int r = 0; r < ROWS; ++r) {
        const int8_t* xr = x + r * WIDTH;
        int32_t sum = 0, sq = 0;
        for (int c = 0; c < WIDTH; ++c) sum += xr[c];
        const int32_t mean = sum >> LOG2_WIDTH;
        for (int c = 0; c < WIDTH; ++c) sq += (xr[c] - mean) * (xr[c] - mean);
        uint32_t sd = isqrt((uint32_t)sq >> LOG2_WIDTH);
        const int32_t inv = (16 << 16) / (int32_t)(sd != 0 ? sd : 1);
        for (int c = 0; c < WIDTH; ++c) {
            int32_t v = (((xr[c] - mean) * inv) >> 16) * gain[c] + (beta[c] << 5);
            v = (v >> 5) + ((v >> 4) & 1);
            if (y[r * WIDTH + c] != (v > 127 ? 127 : v < -128 ? -128 : v)) asm volatile("ebreak");
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
