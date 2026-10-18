// Sim bench: runs one kernel call per halt on the Coral RTL model and times
// it with mcycle. kernel_bench.py (cocotb) writes the inputs into `arena`
// and the call into `bench_args`, runs to halt, and reads back the outputs
// and `bench_cycles`. Everything lives in .data so it survives between runs
// (chunked flash keeps its m/l/acc state in the arena across blocks).
//
// bench_args layout: [0] = Op, then op-specific words. Pointers are byte
// offsets into the arena; floats are passed as their IEEE-754 bits.
//
// BENCH_GROUP selects which kernels this ELF links, to keep each one inside
// the 8 KB ITCM: 0 = matmul, 1 = attention, 2 = pointwise + softmax,
// 3 = microbenchmarks.
#include <stddef.h>
#include <stdint.h>

#include "params.h"

#if BENCH_GROUP == 0
#include "matmul.h"
#include "matmul_rvv.h"
#include "matmul_vme.h"
#include "vme_helpers.h"
#elif BENCH_GROUP == 1
#include "attention_flash.h"
#include "attention_flash_chunk.h"
#include "attention_serial.h"
#elif BENCH_GROUP == 2
#include "gelu.h"
#include "layernorm.h"
#include "online_softmax.h"
#include "residual.h"
#endif

enum Op : uint32_t {
    kMatmulVmeU8 = 1,
    kMatmulVmeS8U8,
    kMatmulRvv,
    kMatmulRvvS8U8,
    kMatmulDispatch,
    kAttentionSerial = 10,
    kAttentionFlash,
    kAttentionFlashChunk,
    kAttentionFlashFinish,
    kSoftmax = 20,
    kGelu,
    kLayernorm,
    kResidual,
    kLoadChainScalar = 30,
    kLoadChainVector,
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

#if BENCH_GROUP == 3
// Dependent-load chains: each load's address is the previous load's result.
// The harness fills the arena so every word points to itself, so the cycles
// per iteration are the load-to-use latency (vector: vle32 + vmv.x.s).
void load_chain_scalar(uint32_t iters, const uint32_t* p) {
    for (uint32_t i = 0; i < iters; ++i) {
        asm volatile("lw %0, 0(%0)\n"
                     "lw %0, 0(%0)\n"
                     "lw %0, 0(%0)\n"
                     "lw %0, 0(%0)\n"
                     : "+r"(p)
                     :
                     : "memory");
    }
}
void load_chain_vector(uint32_t iters, const uint32_t* p) {
    asm volatile("vsetivli zero, 4, e32, m1, ta, ma" ::: "memory");
    for (uint32_t i = 0; i < iters; ++i) {
        asm volatile("vle32.v v1, (%0)\n"
                     "vmv.x.s %0, v1\n"
                     "vle32.v v1, (%0)\n"
                     "vmv.x.s %0, v1\n"
                     : "+r"(p)
                     :
                     : "v1", "memory");
    }
}
#endif

void run() {
    switch (static_cast<Op>(arg(0))) {
#if BENCH_GROUP == 0
    // args: n, k, m, a, b, c [, signed_a]
    case kMatmulVmeU8:
        matmul_vme_u8(arg(1), arg(2), arg(3), at<uint8_t>(4), at<uint8_t>(5),
                      at<int32_t>(6));
        break;
    case kMatmulVmeS8U8:
        matmul_vme_s8u8(arg(1), arg(2), arg(3), at<int8_t>(4), at<uint8_t>(5),
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
#elif BENCH_GROUP == 1
    // args: nq, nk, d, query_start, window, q, kt, v, out, k_zero, v_zero,
    //       score_scale, v_scale, out_scale
    case kAttentionSerial:
        attention_serial(arg(1), arg(2), arg(3), arg(4), arg(5), at<int8_t>(6),
                         at<uint8_t>(7), static_cast<int32_t>(arg(10)),
                         at<uint8_t>(8), static_cast<int32_t>(arg(11)),
                         arg_f(12), arg_f(13), arg_f(14), at<int8_t>(9));
        break;
    case kAttentionFlash:
        attention_flash(arg(1), arg(2), arg(3), arg(4), arg(5), at<int8_t>(6),
                        at<uint8_t>(7), static_cast<int32_t>(arg(10)),
                        at<uint8_t>(8), static_cast<int32_t>(arg(11)),
                        arg_f(12), arg_f(13), arg_f(14), at<int8_t>(9));
        break;
    // args: nq, d, query_start, window, kv_start, kv_len, q, kt, v, m, l,
    //       acc, k_zero, score_scale
    case kAttentionFlashChunk:
        attention_flash_chunk(
            arg(1), arg(2), arg(3), arg(4), arg(5), arg(6), at<int8_t>(7),
            at<uint8_t>(8), static_cast<int32_t>(arg(13)), at<uint8_t>(9),
            arg_f(14), at<float>(10), at<float>(11), at<float>(12));
        break;
    // args: nq, d, l, acc, out, v_zero, v_scale, out_scale
    case kAttentionFlashFinish:
        attention_flash_finish(arg(1), arg(2), at<float>(3), at<float>(4),
                               static_cast<int32_t>(arg(6)), arg_f(7), arg_f(8),
                               at<int8_t>(5));
        break;
#elif BENCH_GROUP == 2
    // args: lo, hi, scores, probs, score_scale
    case kSoftmax:
        online_softmax(at<int32_t>(3), arg(1), arg(2), arg_f(5),
                       at<uint8_t>(4));
        break;
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
#elif BENCH_GROUP == 3
    // args: iters, p
    case kLoadChainScalar:
        load_chain_scalar(arg(1), at<uint32_t>(2));
        break;
    case kLoadChainVector:
        load_chain_vector(arg(1), at<uint32_t>(2));
        break;
#endif
    default:
        break;
    }
}

} // namespace

int main() {
#if BENCH_GROUP == 0
    vme::enable_matrix_state(); // the CRT turns on FP and vector, not matrix
#elif BENCH_GROUP == 2
    if (arg(0) == kGelu) {
        gelu_table(arg_f(5), arg_f(6), at<int8_t>(3)); // setup, not timed
    }
#endif
    const uint32_t start = mcycle();
    run();
    bench_cycles = mcycle() - start;
    return 0;
}
