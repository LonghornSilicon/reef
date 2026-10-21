// Sim bench: runs one kernel call per halt on the Coral RTL model and times
// it with mcycle. bench.py (cocotb) writes the inputs into `arena` and the
// call into `bench_args`, runs to halt, and reads back the outputs and
// `bench_cycles`. Everything lives in .data so it survives between runs
// (block-flash attention keeps its m/l/acc state in the arena across blocks).
//
// bench_args layout: [0] = Op, then op-specific words. Pointers are byte
// offsets into the arena; floats are passed as their IEEE-754 bits.
//
// BENCH_GROUP selects which kernels this ELF links, to keep each one inside
// the 8 KB ITCM: 0 = matmul + attention (matrix engine), 1 = pointwise.
#include <stddef.h>
#include <stdint.h>

#include "helpers.h"
#include "kernels.h"

// Op values are kept from earlier ELFs so old call records stay comparable.
enum Op : uint32_t {
    kVmeMatmulU8 = 1,
    kVmeMatmulS8U8 = 2,
    kMatmulRvv = 3,
    kMatmulRvvS8U8 = 4,
    kMatmulDispatch = 5,
    kAttentionFlashFinish = 13,
    kVmeAttentionBlock = 14,
    kGelu = 21,
    kLayernorm = 22,
    kResidual = 23,
    kSoftmaxLutTable = 25,
};

constexpr size_t kArenaBytes = 20 * 1024;

extern "C" {
uint8_t arena[kArenaBytes] __attribute__((section(".data"), aligned(16)));
volatile uint32_t bench_args[24] __attribute__((section(".data")));
volatile uint32_t bench_cycles __attribute__((section(".data")));
}

namespace {

uint32_t arg(size_t i) { return bench_args[i]; }
[[maybe_unused]] float arg_f(size_t i) {
    const uint32_t bits = bench_args[i];
    float value = 0;
    __builtin_memcpy(&value, &bits, sizeof(value));
    return value;
}
template <typename T> T* at(size_t i) {
    return reinterpret_cast<T*>(&arena[bench_args[i]]);
}

uint32_t mcycle() {
    uint32_t value = 0;
    asm volatile("csrr %0, mcycle" : "=r"(value));
    return value;
}

// Runs the call in bench_args; false if this ELF doesn't have that op.
bool run() {
    switch (static_cast<Op>(arg(0))) {
#if BENCH_GROUP == 0
    // args: n, k, m, a, b, c [, signed_a]
    case kVmeMatmulU8:
        vme_matmul_u8(arg(1), arg(2), arg(3), at<uint8_t>(4), at<uint8_t>(5),
                      at<int32_t>(6));
        break;
    case kVmeMatmulS8U8:
        vme_matmul_s8u8(arg(1), arg(2), arg(3), at<int8_t>(4), at<uint8_t>(5),
                        at<int32_t>(6));
        break;
    case kMatmulRvv:
        matmul_rvv(arg(1), arg(2), arg(3), at<uint8_t>(4), at<uint8_t>(5),
                   at<uint32_t>(6));
        break;
    case kMatmulRvvS8U8:
        matmul_rvv_s8u8(arg(1), arg(2), arg(3), at<int8_t>(4), at<uint8_t>(5),
                        at<int32_t>(6));
        break;
    case kMatmulDispatch:
        matmul(arg(1), arg(2), arg(3), at<uint8_t>(4), arg(7) != 0,
               at<uint8_t>(5), at<int32_t>(6));
        break;
    // args: nq, d, query_start, window, kv_start, kv_len, q, kt, v, m, l,
    //       acc, k_zero, score_scale, scratch, exp_table
    case kVmeAttentionBlock:
        vme_attention_block(arg(1), arg(2), arg(3), arg(4), arg(5), arg(6),
                            at<int8_t>(7), at<uint8_t>(8),
                            static_cast<int32_t>(arg(13)), at<uint8_t>(9),
                            arg_f(14), at<uint16_t>(16), at<int32_t>(15),
                            at<float>(10), at<float>(11), at<float>(12));
        break;
    // args: nq, d, l, acc, out, v_zero, v_scale, out_scale
    case kAttentionFlashFinish:
        attention_flash_finish(arg(1), arg(2), at<float>(3), at<float>(4),
                               static_cast<int32_t>(arg(6)), arg_f(7), arg_f(8),
                               at<int8_t>(5));
        break;
    // args: score_scale, table: the once-per-layer exp table build
    case kSoftmaxLutTable:
        softmax_lut_table(arg_f(1), at<uint16_t>(2));
        break;
#elif BENCH_GROUP == 1
    // args: n, x, table, y, in_scale, out_scale (table built untimed below)
    case kGelu:
        gelu(arg(1), at<int8_t>(2), at<int8_t>(3), at<int8_t>(4));
        break;
    // args: rows, d, x, gamma, beta, out, x_scale, gamma_scale, beta_scale,
    //       out_scale
    case kLayernorm:
        layernorm(arg(1), arg(2), at<int8_t>(3), arg_f(7), at<int8_t>(4),
                  arg_f(8), at<int8_t>(5), arg_f(9), arg_f(10), at<int8_t>(6));
        break;
    // args: n, a, b, out, a_scale, b_scale, out_scale
    case kResidual:
        residual_add(arg(1), at<int8_t>(2), arg_f(5), at<int8_t>(3), arg_f(6),
                     arg_f(7), at<int8_t>(4));
        break;
#endif
    default:
        return false;
    }
    return true;
}

} // namespace

int main() {
#if BENCH_GROUP == 0
    vme::enable_matrix_state(); // the CRT turns on FP and vector, not matrix
#elif BENCH_GROUP == 1
    if (arg(0) == kGelu) {
        gelu_table(arg_f(5), arg_f(6), at<int8_t>(3)); // setup, not timed
    }
#endif
    const uint32_t start = mcycle();
    const bool ran = run();
    // an op this ELF doesn't link reports all-ones, which the harness rejects
    bench_cycles = ran ? mcycle() - start : 0xFFFFFFFFU;
    return 0;
}
