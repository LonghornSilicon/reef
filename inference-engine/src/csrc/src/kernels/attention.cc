// Attention (operator/attention.hpp) on Coral M3, ready for reef_perf: one
// causal head over T positions of width D.
//   scores = q . k^T                  int8 x int8 -> int32
//   p      = softmax(scores), causal  uint8, each row sums to ~255
//   out    = p . v >> 8               uint8 x int8 -> int8
// The 1/sqrt(D) scale is folded into SCORE_SHIFT and the exp table.
#include <riscv_vector.h>
#include <stdint.h>

#define T 16
#define D 64
#define SCORE_SHIFT 10

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

int8_t q[T * D];
int8_t kt[D * T];  // K^T, so each row of kt is contiguous over positions
int8_t v[T * D];
int32_t scores[T * T];
uint8_t probs[T * T];  // zero above the diagonal
int32_t acc[T * D];
int8_t out[T * D];

// c[n x m] = a[n x k] . b[k x m], 16 columns per strip
template <typename A>
void matmul(int n, int k, int m, const A* a, const int8_t* b, int32_t* c) {
    for (int col = 0; col < m;) {
        size_t vl = __riscv_vsetvl_e8m1(m - col);
        for (int i = 0; i < n; ++i) {
            vint32m4_t sum = __riscv_vmv_v_x_i32m4(0, vl);
            for (int j = 0; j < k; ++j) {
                vint16m2_t row = __riscv_vsext_vf2_i16m2(
                    __riscv_vle8_v_i8m1(b + j * m + col, vl), vl);
                sum = __riscv_vwmacc_vx_i32m4(sum, a[i * k + j], row, vl);
            }
            __riscv_vse32_v_i32m4(c + i * m + col, sum, vl);
        }
        col += vl;
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

void attention() {
    matmul(T, D, T, q, kt, scores);
    for (int i = 0; i < T; ++i) {
        softmax(i + 1, scores + i * T, SCORE_SHIFT, exp_lut, probs + i * T);  // keys 0..i
    }
    matmul(T, T, D, probs, v, acc);
    for (int i = 0; i < T * D;) {
        size_t vl = __riscv_vsetvl_e8m1(T * D - i);
        vint16m2_t h = __riscv_vnclip_wx_i16m2(__riscv_vle32_v_i32m4(acc + i, vl), 8,
                                               __RISCV_VXRM_RNU, vl);
        __riscv_vse8_v_i8m1(out + i, __riscv_vnclip_wx_i8m1(h, 0, __RISCV_VXRM_RNU, vl), vl);
        i += vl;
    }
}

uint32_t seed = 1;
int8_t rnd() {
    seed = seed * 1664525u + 1013904223u;
    return (int8_t)(seed >> 24);
}

int main() {
    for (int i = 0; i < T * D; ++i) q[i] = rnd();
    for (int i = 0; i < D * T; ++i) kt[i] = rnd();
    for (int i = 0; i < T * D; ++i) v[i] = rnd();
    attention();
#ifdef CHECK  // scalar reference; a mismatch traps reef_perf
    for (int i = 0; i < T; ++i) {
        int32_t s[T], m;
        for (int j = 0; j <= i; ++j) {
            s[j] = 0;
            for (int c = 0; c < D; ++c) s[j] += q[i * D + c] * kt[c * T + j];
            m = j == 0 || s[j] > m ? s[j] : m;
        }
        uint32_t e[T], sum = 0;
        for (int j = 0; j <= i; ++j) {
            uint32_t d = (uint32_t)((m - s[j]) >> SCORE_SHIFT);
            sum += e[j] = exp_lut[d < 255 ? d : 255];
        }
        uint32_t rr = (255u << 16) / sum;
        int32_t p[T] = {0};
        for (int j = 0; j <= i; ++j) {
            p[j] = (e[j] * rr >> 16) + ((e[j] * rr >> 15) & 1);
            p[j] = p[j] > 255 ? 255 : p[j];
        }
        for (int c = 0; c < D; ++c) {
            int32_t a = 0;
            for (int j = 0; j < T; ++j) a += p[j] * v[j * D + c];
            a = (a >> 8) + ((a >> 7) & 1);
            if (out[i * D + c] != (a > 127 ? 127 : a < -128 ? -128 : a)) asm volatile("ebreak");
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
