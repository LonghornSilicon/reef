// MLP (operator/mlp.hpp) on Coral M3, ready for reef_perf:
//   h = gelu(x . W_fc^T + b_fc)    D -> H
//   y = h . W_proj^T + b_proj      H -> D
// int8 between the steps; GELU is the host-precomputed table (scale 0.05).
#include <riscv_vector.h>
#include <stdint.h>

#define ROWS 16
#define D 32
#define H 128
#define SHIFT 8

// gelu_lut[i] = int8(gelu((i - 128) * 0.05) / 0.05), rounded half away from 0
const int8_t gelu_lut[256] = {
       0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
       0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
       0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
       0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
       0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
       0,    0,    0,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -2,   -2,
      -2,   -2,   -2,   -2,   -2,   -2,   -3,   -3,   -3,   -3,   -3,   -3,   -3,   -3,   -3,   -3,
      -3,   -3,   -3,   -3,   -3,   -3,   -3,   -3,   -3,   -3,   -2,   -2,   -2,   -1,   -1,    0,
       0,    1,    1,    2,    2,    3,    4,    4,    5,    6,    7,    8,    9,   10,   11,   12,
      13,   14,   15,   16,   17,   18,   19,   20,   21,   22,   23,   25,   26,   27,   28,   29,
      30,   31,   32,   34,   35,   36,   37,   38,   39,   40,   41,   42,   43,   44,   46,   47,
      48,   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,   59,   60,   61,   62,   63,
      64,   65,   66,   67,   68,   69,   70,   71,   72,   73,   74,   75,   76,   77,   78,   79,
      80,   81,   82,   83,   84,   85,   86,   87,   88,   89,   90,   91,   92,   93,   94,   95,
      96,   97,   98,   99,  100,  101,  102,  103,  104,  105,  106,  107,  108,  109,  110,  111,
     112,  113,  114,  115,  116,  117,  118,  119,  120,  121,  122,  123,  124,  125,  126,  127,
};

int8_t x[ROWS * D];
int8_t fc_wt[D * H];  // W_fc^T
int32_t fc_bias[H];
int8_t proj_wt[H * D];  // W_proj^T
int32_t proj_bias[D];
int8_t h[ROWS * H];
int8_t y[ROWS * D];

// int32 -> int8: round half up after >> shift, then saturate
vint8m1_t narrow(vint32m4_t v, int shift, size_t vl) {
    vint16m2_t n = __riscv_vnclip_wx_i16m2(v, shift, __RISCV_VXRM_RNU, vl);
    return __riscv_vnclip_wx_i8m1(n, 0, __RISCV_VXRM_RNU, vl);
}

// y = x . W^T + bias as in linear.cc; gelu applies the table on the way out
void linear(int rows, int in, int out, const int8_t* x, const int8_t* wt,
            const int32_t* bias, int shift, bool gelu, int8_t* y) {
    for (int c = 0; c < out;) {
        size_t vl = __riscv_vsetvl_e8m1(out - c);
        for (int r = 0; r < rows; ++r) {
            vint32m4_t acc = __riscv_vle32_v_i32m4(bias + c, vl);
            for (int k = 0; k < in; ++k) {
                vint16m2_t w = __riscv_vsext_vf2_i16m2(
                    __riscv_vle8_v_i8m1(wt + k * out + c, vl), vl);
                acc = __riscv_vwmacc_vx_i32m4(acc, x[r * in + k], w, vl);
            }
            vint8m1_t v = narrow(acc, shift, vl);
            if (gelu) {
                vuint8m1_t idx = __riscv_vxor_vx_u8m1(__riscv_vreinterpret_v_i8m1_u8m1(v), 0x80, vl);
                v = __riscv_vluxei8_v_i8m1(gelu_lut, idx, vl);
            }
            __riscv_vse8_v_i8m1(y + r * out + c, v, vl);
        }
        c += vl;
    }
}

void mlp() {
    linear(ROWS, D, H, x, fc_wt, fc_bias, SHIFT, true, h);
    linear(ROWS, H, D, h, proj_wt, proj_bias, SHIFT, false, y);
}

uint32_t seed = 1;
int8_t rnd() {
    seed = seed * 1664525u + 1013904223u;
    return (int8_t)(seed >> 24);
}

#ifdef CHECK
int8_t ref(const int8_t* x, int k, const int8_t* wt, int out, int c, int32_t b) {
    int32_t acc = b;
    for (int j = 0; j < k; ++j) acc += x[j] * wt[j * out + c];
    acc = (acc >> SHIFT) + ((acc >> (SHIFT - 1)) & 1);
    return acc > 127 ? 127 : acc < -128 ? -128 : acc;
}
#endif

int main() {
    for (int i = 0; i < ROWS * D; ++i) x[i] = rnd();
    for (int i = 0; i < D * H; ++i) fc_wt[i] = rnd();
    for (int i = 0; i < H; ++i) fc_bias[i] = rnd() * 16;
    for (int i = 0; i < H * D; ++i) proj_wt[i] = rnd();
    for (int i = 0; i < D; ++i) proj_bias[i] = rnd() * 16;
    mlp();
#ifdef CHECK  // scalar reference; a mismatch traps reef_perf
    for (int r = 0; r < ROWS; ++r) {
        int8_t hr[H];
        for (int c = 0; c < H; ++c) {
            hr[c] = gelu_lut[ref(x + r * D, D, fc_wt, H, c, fc_bias[c]) + 128];
        }
        for (int c = 0; c < D; ++c) {
            if (y[r * D + c] != ref(hr, H, proj_wt, D, c, proj_bias[c])) asm volatile("ebreak");
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
