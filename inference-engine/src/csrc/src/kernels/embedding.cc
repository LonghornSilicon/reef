// Embedding (operator/embedding.hpp) on Coral M3, ready for reef_perf:
//   y[t] = tokens[ids[t]] + positions[t]   int8, saturating
// Both tables are taken to share one scale, so no rescale is needed.
#include <riscv_vector.h>
#include <stdint.h>

#define LEN 16
#define WIDTH 64
#define VOCAB 64

uint32_t ids[LEN];
int8_t tokens[VOCAB * WIDTH];
int8_t positions[LEN * WIDTH];
int8_t y[LEN * WIDTH];

void embedding(int len, int width, const uint32_t* ids, const int8_t* tokens,
               const int8_t* positions, int8_t* y) {
    for (int t = 0; t < len; ++t) {
        const int8_t* row = tokens + ids[t] * width;
        for (int c = 0; c < width;) {
            size_t vl = __riscv_vsetvl_e8m1(width - c);
            vint8m1_t sum = __riscv_vsadd_vv_i8m1(__riscv_vle8_v_i8m1(row + c, vl),
                                                  __riscv_vle8_v_i8m1(positions + t * width + c, vl), vl);
            __riscv_vse8_v_i8m1(y + t * width + c, sum, vl);
            c += vl;
        }
    }
}

uint32_t seed = 1;
int8_t rnd() {
    seed = seed * 1664525u + 1013904223u;
    return (int8_t)(seed >> 24);
}

int main() {
    for (int t = 0; t < LEN; ++t) ids[t] = (uint8_t)rnd() % VOCAB;
    for (int i = 0; i < VOCAB * WIDTH; ++i) tokens[i] = rnd();
    for (int i = 0; i < LEN * WIDTH; ++i) positions[i] = rnd();
    embedding(LEN, WIDTH, ids, tokens, positions, y);
#ifdef CHECK  // scalar reference; a mismatch traps reef_perf
    for (int t = 0; t < LEN; ++t) {
        for (int c = 0; c < WIDTH; ++c) {
            int s = tokens[ids[t] * WIDTH + c] + positions[t * WIDTH + c];
            if (y[t * WIDTH + c] != (s > 127 ? 127 : s < -128 ? -128 : s)) asm volatile("ebreak");
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
