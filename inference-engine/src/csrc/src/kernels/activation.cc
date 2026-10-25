// Activations (operator/activation.hpp) on Coral M3, ready for reef_perf:
//   gelu:    int8 -> int8 through a 256-entry table (in/out scale 0.05)
//   softmax: int32 scores -> uint8 probabilities (sum ~255) per row
// Both tables are precomputed on the host, so the core only looks them up.
#include <riscv_vector.h>
#include <stdint.h>

#define N 1024  // gelu elements
#define ROWS 16 // softmax rows of COLS scores
#define COLS 64
#define SCORE_SHIFT 4

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

// exp_lut[d] = round(32768 * exp(-d / 16)); zero from d = 178 on
const uint16_t exp_lut[256] = {
    32768, 30783, 28918, 27166, 25520, 23974, 22521, 21157, 19875, 18671, 17539, 16477,
    15479, 14541, 13660, 12832, 12055, 11324, 10638, 9994, 9388, 8819, 8285, 7783,
    7312, 6869, 6452, 6061, 5694, 5349, 5025, 4721, 4435, 4166, 3914, 3676,
    3454, 3244, 3048, 2863, 2690, 2527, 2374, 2230, 2095, 1968, 1849, 1737,
    1631, 1533, 1440, 1352, 1271, 1194, 1121, 1053, 990, 930, 873, 820,
    771, 724, 680, 639, 600, 564, 530, 498, 467, 439, 412, 387,
    364, 342, 321, 302, 283, 266, 250, 235, 221, 207, 195, 183,
    172, 162, 152, 143, 134, 126, 118, 111, 104, 98, 92, 86,
    81, 76, 72, 67, 63, 59, 56, 52, 49, 46, 43, 41,
    38, 36, 34, 32, 30, 28, 26, 25, 23, 22, 21, 19,
    18, 17, 16, 15, 14, 13, 12, 12, 11, 10, 10, 9,
    9, 8, 8, 7, 7, 6, 6, 6, 5, 5, 5, 4,
    4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2,
    2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
};

int8_t x[N];
int8_t y[N];
int32_t scores[ROWS * COLS];
uint8_t probs[ROWS * COLS];

// y[i] = gelu_lut[x[i] + 128], 16 bytes per vector strip
void gelu(int n, const int8_t* x, const int8_t* lut, int8_t* y) {
    for (int i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e8m1(n - i);
        vuint8m1_t v = __riscv_vreinterpret_v_i8m1_u8m1(__riscv_vle8_v_i8m1(x + i, vl));
        vuint8m1_t idx = __riscv_vxor_vx_u8m1(v, 0x80, vl);  // x + 128
        __riscv_vse8_v_i8m1(y + i, __riscv_vluxei8_v_i8m1(lut, idx, vl), vl);
        i += vl;
    }
}

// e = exp_lut[min((m - s) >> shift, 255)]
vuint32m4_t exp_lookup(const int32_t* s, int32_t m, int shift, const uint16_t* lut,
                       size_t vl) {
    vint32m4_t d = __riscv_vsra_vx_i32m4(
        __riscv_vrsub_vx_i32m4(__riscv_vle32_v_i32m4(s, vl), m, vl), shift, vl);
    vuint32m4_t idx = __riscv_vminu_vx_u32m4(__riscv_vreinterpret_v_i32m4_u32m4(d), 255, vl);
    idx = __riscv_vsll_vx_u32m4(idx, 1, vl);  // byte offset of a uint16 entry
    return __riscv_vzext_vf2_u32m4(__riscv_vluxei32_v_u16m2(lut, idx, vl), vl);
}

// p[j] = round(255 * e[j] / sum(e)): row max, exp sum, one divide, scale
void softmax(int n, const int32_t* s, int shift, const uint16_t* lut, uint8_t* p) {
    size_t vl;
    int32_t m = s[0];
    for (int j = 0; j < n; j += vl) {
        vl = __riscv_vsetvl_e32m4(n - j);
        m = __riscv_vmv_x_s_i32m1_i32(__riscv_vredmax_vs_i32m4_i32m1(
            __riscv_vle32_v_i32m4(s + j, vl), __riscv_vmv_s_x_i32m1(m, vl), vl));
    }
    uint32_t sum = 0;
    for (int j = 0; j < n; j += vl) {
        vl = __riscv_vsetvl_e32m4(n - j);
        sum = __riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(
            exp_lookup(s + j, m, shift, lut, vl), __riscv_vmv_s_x_u32m1(sum, vl), vl));
    }
    uint32_t r = (255u << 16) / sum;  // sum >= 32768: the max's exp
    for (int j = 0; j < n; j += vl) {
        vl = __riscv_vsetvl_e32m4(n - j);
        vuint32m4_t pr = __riscv_vmul_vx_u32m4(exp_lookup(s + j, m, shift, lut, vl), r, vl);
        vuint16m2_t h = __riscv_vnclipu_wx_u16m2(pr, 16, __RISCV_VXRM_RNU, vl);
        __riscv_vse8_v_u8m1(p + j, __riscv_vnclipu_wx_u8m1(h, 0, __RISCV_VXRM_RNU, vl), vl);
    }
}

uint32_t seed = 1;
int8_t rnd() {
    seed = seed * 1664525u + 1013904223u;
    return (int8_t)(seed >> 24);
}

int main() {
    for (int i = 0; i < N; ++i) x[i] = rnd();
    for (int i = 0; i < ROWS * COLS; ++i) scores[i] = rnd() * 16;
    gelu(N, x, gelu_lut, y);
    for (int r = 0; r < ROWS; ++r) {
        softmax(COLS, scores + r * COLS, SCORE_SHIFT, exp_lut, probs + r * COLS);
    }
#ifdef CHECK  // scalar reference; a mismatch traps reef_perf
    for (int i = 0; i < N; ++i) {
        if (y[i] != gelu_lut[x[i] + 128]) asm volatile("ebreak");
    }
    for (int r = 0; r < ROWS; ++r) {
        const int32_t* s = scores + r * COLS;
        int32_t m = s[0];
        for (int j = 1; j < COLS; ++j) m = s[j] > m ? s[j] : m;
        uint32_t e[COLS], sum = 0;
        for (int j = 0; j < COLS; ++j) {
            uint32_t d = (uint32_t)((m - s[j]) >> SCORE_SHIFT);
            sum += e[j] = exp_lut[d < 255 ? d : 255];
        }
        uint32_t rr = (255u << 16) / sum;
        for (int j = 0; j < COLS; ++j) {
            uint32_t p = (e[j] * rr >> 16) + ((e[j] * rr >> 15) & 1);
            if (probs[r * COLS + j] != (p > 255 ? 255 : p)) asm volatile("ebreak");
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
