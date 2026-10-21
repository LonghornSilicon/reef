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
// the 8 KB ITCM. One ELF per operator: 0 = linear, 1 = attention, 2 =
// pointwise (activation, plus LayerNorm and residual add), 3 = mlp.
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
    kLinear = 6,
    kAttentionFlashFinish = 13,
    kVmeAttentionBlock = 14,
    kGelu = 21,
    kLayernorm = 22,
    kResidual = 23,
    kSoftmaxLut = 24,
    kSoftmaxLutTable = 25,
    kMlpBlock = 40,
    kMlpFinish = 41,
};

// An argument word meaning "no buffer" (e.g. a linear without bias).
constexpr uint32_t kNull = 0xFFFFFFFFU;

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
[[maybe_unused]] const int32_t* bias_at(size_t i) {
    return bench_args[i] == kNull ? nullptr : at<int32_t>(i);
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
    // args: rows, in, out, x, wt, w_zero, bias (or kNull), acc_scale,
    //       out_scale, scratch, y
    case kLinear:
        linear(arg(1), arg(2), arg(3), at<int8_t>(4), at<uint8_t>(5),
               static_cast<int32_t>(arg(6)), bias_at(7), arg_f(8), arg_f(9),
               at<int32_t>(10), at<int8_t>(11));
        break;
#elif BENCH_GROUP == 1
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
#elif BENCH_GROUP == 2
    // args: n, x, table, y, in_scale, out_scale (table built untimed below)
    case kGelu:
        gelu(arg(1), at<int8_t>(2), at<int8_t>(3), at<int8_t>(4));
        break;
    // args: lo, hi, scores, probs, score_scale, table (built untimed below)
    case kSoftmaxLut:
        softmax_lut(at<int32_t>(3), arg(1), arg(2), at<uint16_t>(6),
                    at<uint8_t>(4));
        break;
    // args: score_scale, table: the once-per-layer exp table build
    case kSoftmaxLutTable:
        softmax_lut_table(arg_f(1), at<uint16_t>(2));
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
#elif BENCH_GROUP == 3
    // args: rows, d, chunk, x, fc_wt, fc_zero, fc_bias, fc_scale,
    //       gelu_table, proj_wt, proj_zero, scratch, acc
    case kMlpBlock:
        mlp_block(arg(1), arg(2), arg(3), at<int8_t>(4), at<uint8_t>(5),
                  static_cast<int32_t>(arg(6)), at<int32_t>(7), arg_f(8),
                  at<int8_t>(9), at<uint8_t>(10), static_cast<int32_t>(arg(11)),
                  at<int32_t>(12), at<int32_t>(13));
        break;
    // args: rows, d, acc, proj_bias, acc_scale, out_scale, y
    case kMlpFinish:
        mlp_finish(arg(1), arg(2), at<int32_t>(3), at<int32_t>(4), arg_f(5),
                   arg_f(6), at<int8_t>(7));
        break;
#endif
    default:
        return false;
    }
    return true;
}

} // namespace

int main() {
#if BENCH_GROUP == 2
    if (arg(0) == kGelu) {
        gelu_table(arg_f(5), arg_f(6), at<int8_t>(3)); // setup, not timed
    } else if (arg(0) == kSoftmaxLut) {
        softmax_lut_table(arg_f(5), at<uint16_t>(6)); // setup, not timed
    }
#else
    vme::enable_matrix_state(); // the CRT turns on FP and vector, not matrix
#endif
    const uint32_t start = mcycle();
    const bool ran = run();
    // an op this ELF doesn't link reports all-ones, which the harness rejects
    bench_cycles = ran ? mcycle() - start : 0xFFFFFFFFU;
    return 0;
}
