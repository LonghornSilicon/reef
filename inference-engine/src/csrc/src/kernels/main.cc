// Runs one kernel on Coral M3: the RTL testbench or reef_perf. Pick it with
// -DKERNEL_LINEAR, _ATTENTION, _MLP, _GELU, _SOFTMAX, _LAYERNORM,
// _RESIDUAL or _EMBEDDING; see README.md for the build.
//
// Passing values: every input, output and scale is a named global in .data,
// so a testbench can find it in the ELF's symbol table.
// - reef_perf has no host, so main() fills the inputs from a fixed seed.
// - A testbench that writes its own inputs sets fill_inputs = 0 first.
//   (.bss is zeroed at startup, which is why everything is in .data.)
// After the run, `cycles` holds mcycle around the kernel calls only, not the
// one-time table builds.
#include "kernels.h"

#ifdef __riscv
#define DATA __attribute__((section(".data"), aligned(16)))
#else // host build: no ELF sections to choose
#define DATA __attribute__((aligned(16)))
#endif

// Shapes; override with -D. One GPT-2 head is D = 64; decode is ROWS = 1.
#ifndef ROWS
#define ROWS 16
#endif
#ifndef D
#define D 64
#endif
#ifndef COLS // linear's out, attention's kv_len, MLP's chunk, vocab rows
#define COLS 64
#endif

volatile uint32_t fill_inputs DATA = 1;
volatile uint32_t cycles DATA = 0;

namespace {

uint32_t read_mcycle() {
    uint32_t value = 0;
#ifdef __riscv // 0 in a host build (-DRVV_EMULATE -DVME_EMULATE)
    asm volatile("csrr %0, mcycle" : "=r"(value));
#endif
    return value;
}

// Fixed-seed bytes, so every run and every simulator sees the same inputs.
void fill(void* buffer, size_t bytes, uint32_t seed) {
    auto* out = static_cast<uint8_t*>(buffer);
    for (size_t i = 0; i < bytes; ++i) {
        seed = (seed * 1664525U) + 1013904223U;
        out[i] = static_cast<uint8_t>(seed >> 24U);
    }
}

} // namespace

#if defined(KERNEL_LINEAR)
int8_t x[ROWS * D] DATA;
uint8_t wt[D * COLS] DATA;
int32_t bias[COLS] DATA;
int32_t scratch[ROWS * COLS] DATA;
int8_t y[ROWS * COLS] DATA;
int32_t w_zero DATA = 128;
float acc_scale DATA = 1e-4F;
float out_scale DATA = 0.05F;

void run() {
    if (fill_inputs != 0) {
        fill(x, sizeof(x), 1);
        fill(wt, sizeof(wt), 2);
        fill(bias, sizeof(bias), 3);
    }
    const uint32_t start = read_mcycle();
    linear(ROWS, D, COLS, x, wt, w_zero, bias, acc_scale, out_scale, scratch,
           y);
    cycles = read_mcycle() - start;
}

#elif defined(KERNEL_ATTENTION)
// One block of COLS keys; query rows sit at the end of it, so the causal
// mask leaves the last row every key and the first row COLS - ROWS + 1.
constexpr size_t kScratch = (ROWS * (COLS > D ? COLS : D)) + (ROWS * COLS / 4);
int8_t q[ROWS * D] DATA;
uint8_t kt[D * COLS] DATA;
uint8_t v[COLS * D] DATA;
uint16_t exp_table[kSoftmaxLutEntries] DATA;
int32_t scratch[kScratch] DATA;
float m[ROWS] DATA;
float l[ROWS] DATA;
float acc[ROWS * D] DATA;
int8_t out[ROWS * D] DATA;
int32_t k_zero DATA = 128;
int32_t v_zero DATA = 128;
float score_scale DATA = 0.002F;
float v_scale DATA = 0.02F;
float out_scale DATA = 0.02F;

void run() {
    if (fill_inputs != 0) {
        fill(q, sizeof(q), 1);
        fill(kt, sizeof(kt), 2);
        fill(v, sizeof(v), 3);
    }
    softmax_lut_table(score_scale, exp_table);
    for (size_t i = 0; i < ROWS * D; ++i) {
        acc[i] = 0.0F;
    }
    for (float& value : l) {
        value = 0.0F;
    }
    const uint32_t start = read_mcycle();
    vme_attention_block(ROWS, D, /*query_start=*/COLS - ROWS,
                        /*window=*/COLS, /*kv_start=*/0, COLS, q, kt, k_zero, v,
                        score_scale, exp_table, scratch, m, l, acc);
    attention_flash_finish(ROWS, D, l, acc, v_zero, v_scale, out_scale, out);
    cycles = read_mcycle() - start;
}

#elif defined(KERNEL_MLP)
// One chunk of COLS inner columns: c_fc, GeLU, c_proj, then the finish.
constexpr size_t kScratch = (ROWS * (COLS > D ? COLS : D)) + (ROWS * COLS / 4);
int8_t x[ROWS * D] DATA;
uint8_t fc_wt[D * COLS] DATA;
int32_t fc_bias[COLS] DATA;
int8_t gelu_lut[256] DATA;
uint8_t proj_wt[COLS * D] DATA;
int32_t proj_bias[D] DATA;
int32_t scratch[kScratch] DATA;
int32_t acc[ROWS * D] DATA;
int8_t y[ROWS * D] DATA;
int32_t fc_zero DATA = 128;
int32_t proj_zero DATA = 128;
float fc_scale DATA = 1e-3F;
float gelu_in_scale DATA = 0.05F;
float gelu_out_scale DATA = 0.05F;
float acc_scale DATA = 1e-4F;
float out_scale DATA = 0.05F;

void run() {
    if (fill_inputs != 0) {
        fill(x, sizeof(x), 1);
        fill(fc_wt, sizeof(fc_wt), 2);
        fill(fc_bias, sizeof(fc_bias), 3);
        fill(proj_wt, sizeof(proj_wt), 4);
        fill(proj_bias, sizeof(proj_bias), 5);
    }
    gelu_table(gelu_in_scale, gelu_out_scale, gelu_lut);
    for (int32_t& value : acc) {
        value = 0;
    }
    const uint32_t start = read_mcycle();
    mlp_block(ROWS, D, COLS, x, fc_wt, fc_zero, fc_bias, fc_scale, gelu_lut,
              proj_wt, proj_zero, scratch, acc);
    mlp_finish(ROWS, D, acc, proj_bias, acc_scale, out_scale, y);
    cycles = read_mcycle() - start;
}

#elif defined(KERNEL_GELU)
int8_t x[ROWS * D] DATA;
int8_t gelu_lut[256] DATA;
int8_t y[ROWS * D] DATA;
float in_scale DATA = 0.05F;
float out_scale DATA = 0.05F;

void run() {
    if (fill_inputs != 0) {
        fill(x, sizeof(x), 1);
    }
    gelu_table(in_scale, out_scale, gelu_lut);
    const uint32_t start = read_mcycle();
    gelu(ROWS * D, x, gelu_lut, y);
    cycles = read_mcycle() - start;
}

#elif defined(KERNEL_SOFTMAX)
// ROWS score rows of COLS keys, each row's scores in [-128, 127].
int32_t scores[ROWS * COLS] DATA;
uint16_t exp_table[kSoftmaxLutEntries] DATA;
uint8_t probs[ROWS * COLS] DATA;
float score_scale DATA = 0.002F;

void run() {
    if (fill_inputs != 0) {
        int8_t bytes[COLS];
        for (size_t r = 0; r < ROWS; ++r) {
            fill(bytes, sizeof(bytes), r + 1);
            for (size_t c = 0; c < COLS; ++c) {
                scores[(r * COLS) + c] = bytes[c];
            }
        }
    }
    softmax_lut_table(score_scale, exp_table);
    const uint32_t start = read_mcycle();
    for (size_t r = 0; r < ROWS; ++r) {
        softmax_lut(scores + (r * COLS), 0, COLS - 1, exp_table,
                    probs + (r * COLS));
    }
    cycles = read_mcycle() - start;
}

#elif defined(KERNEL_LAYERNORM)
int8_t x[ROWS * D] DATA;
int8_t gamma[D] DATA;
int8_t beta[D] DATA;
int8_t out[ROWS * D] DATA;
float x_scale DATA = 0.05F;
float gamma_scale DATA = 0.01F;
float beta_scale DATA = 0.01F;
float out_scale DATA = 0.05F;

void run() {
    if (fill_inputs != 0) {
        fill(x, sizeof(x), 1);
        fill(gamma, sizeof(gamma), 2);
        fill(beta, sizeof(beta), 3);
    }
    const uint32_t start = read_mcycle();
    layernorm(ROWS, D, x, x_scale, gamma, gamma_scale, beta, beta_scale,
              out_scale, out);
    cycles = read_mcycle() - start;
}

#elif defined(KERNEL_RESIDUAL)
int8_t a[ROWS * D] DATA;
int8_t b[ROWS * D] DATA;
int8_t out[ROWS * D] DATA;
float a_scale DATA = 0.05F;
float b_scale DATA = 0.04F;
float out_scale DATA = 0.06F;

void run() {
    if (fill_inputs != 0) {
        fill(a, sizeof(a), 1);
        fill(b, sizeof(b), 2);
    }
    const uint32_t start = read_mcycle();
    residual_add(ROWS * D, a, a_scale, b, b_scale, out_scale, out);
    cycles = read_mcycle() - start;
}

#elif defined(KERNEL_EMBEDDING)
// ROWS tokens looked up in a COLS-row table slice, then position-added.
uint32_t ids[ROWS] DATA;
int8_t table[COLS * D] DATA;
int8_t positions[ROWS * D] DATA;
int8_t y[ROWS * D] DATA;
uint32_t filled DATA = 0;
float token_scale DATA = 0.05F;
float position_scale DATA = 0.02F;
float out_scale DATA = 0.06F;

void run() {
    if (fill_inputs != 0) {
        fill(table, sizeof(table), 1);
        fill(positions, sizeof(positions), 2);
        fill(ids, sizeof(ids), 3);
        for (uint32_t& id : ids) {
            id %= COLS;
        }
    }
    const uint32_t start = read_mcycle();
    filled = embedding_lookup(ROWS, D, ids, 0, COLS, table, y);
    add_position_embeddings(ROWS, D, y, token_scale, positions, position_scale,
                            0, out_scale, y);
    cycles = read_mcycle() - start;
}

#else
#error "define one KERNEL_<NAME>; see the top of this file"
#endif

int main() {
    run();
    return 0;
}

#ifdef REEF_START
// Startup for builds without Coral's crt (the plain command in README.md):
// set the stack, turn on the FPU and vector unit, run main, then mpause,
// which stops reef_perf and the M3 testbench. reef_perf's ELF loader zeroes
// .bss itself.
asm(R"(
    .section .text.start, "ax"
    .globl _start
_start:
    la    sp, _stack_top
    li    t0, (1 << 13) | (1 << 9)
    csrs  mstatus, t0
    call  main
    .word 0x08000073
1:  j     1b
)");
#endif
