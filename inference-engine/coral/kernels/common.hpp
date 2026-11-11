#pragma once

// Shared by the kernels: DTCM layout, epilogue modes, small vector helpers
// and the kernel entry points. Each program (main.cc, standalone.cc) defines
// `dtcm`.

#include "npu.hpp"

namespace npu {

constexpr uint32_t kChunkRows = 32; // weight rows (of 64 B) per DMA chunk: 2 KB
// KV blocks per attention DMA (about 1 KB).
constexpr uint32_t kKvChunk = kKvBlock < 1024 ? 1024 / kKvBlock : 1;
constexpr int32_t kNegInf = -(1 << 28); // masked score
// Requantization shift (placeholder scale).
constexpr uint32_t kRequantShift = 8;
constexpr uint32_t kExpShift = 4; // score units per exp-table step

// The DTCM: 32 KB minus the mailbox and the stack.
// Activation rows are kBlockTokens bytes apart; vectors are 16 lanes, so a row
// load reads up to 15 bytes past its row (lanes >= kBlockTokens are ignored):
// kRowPad.
constexpr uint32_t kRowPad = 16;
struct Attn {
    int8_t query[kHidden * kBlockTokens];
    int8_t key[kHidden * kBlockTokens];    // new K^T, [feature][token]
    uint8_t value[kBlockTokens * kHidden]; // new V, [token][feature]
    uint8_t kv_buf[2][2][kKvChunk * kKvBlock];
};
struct Softmax {
    uint8_t probs[256];             // P^T [key][token]
    int32_t out_acc[kHeadDim * 16]; // [dim][token]
    int32_t row_max[16];
    int32_t row_sum[16];
};
struct Dtcm {
    int8_t residual[kHidden * kBlockTokens + kRowPad]; // residual stream
    int8_t normed[kHidden * kBlockTokens +
                  kRowPad]; // LayerNorm out, then attention out
    union {
        int8_t mlp[kMlpDim * kBlockTokens + kRowPad]; // MLP inner
        Attn attn;
        int8_t norm_params[2 * kHidden];             // LayerNorm gamma, beta
        int8_t embed_rows[kBlockTokens][2][kHidden]; // token and position rows
    } shared;
    union {
        uint8_t
            weights[2][kChunkRows * 64 + 256]; // weight chunks (+ bias trailer)
        Softmax softmax;
    } scratch;
    int32_t act_sum[16];
    int32_t best_logit[16];
    int32_t best_index[16];
    int8_t gelu_table[256];
    uint8_t exp_table[256];
};
extern Dtcm dtcm; // defined by the program (main.cc, standalone.cc)

enum Mode : uint8_t { kFeatureMajor, kGelu, kResidual, kValueRows, kArgmax };

E2E_INLINE vint32m4_t splat(int32_t value) {
    return __riscv_vmv_v_x_i32m4(value, 16);
}
E2E_INLINE vint32m4_t load_i8(const int8_t* ptr) {
    return __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(ptr, 16), 16);
}
E2E_INLINE vint8m1_t to_i8(vint32m4_t value) {
    value =
        __riscv_vmin_vx_i32m4(__riscv_vmax_vx_i32m4(value, -128, 16), 127, 16);
    return __riscv_vncvt_x_x_w_i8m1(__riscv_vncvt_x_x_w_i16m2(value, 16), 16);
}
// exp(-gap / 16) * 255 from the table, as int32.
E2E_INLINE vint32m4_t exp_lut(vint32m4_t gap) {
    vuint32m4_t index = __riscv_vsrl_vx_u32m4(
        __riscv_vreinterpret_v_i32m4_u32m4(gap), kExpShift, 16);
    index = __riscv_vminu_vx_u32m4(index, 255, 16);
    return __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4_u32m4(
        __riscv_vluxei32_v_u8m1(dtcm.exp_table, index, 16), 16));
}

// mcycle read: reef-perf lists every read's cycle (JSON mcycle_reads), so a
// mark after a kernel times it.
E2E_INLINE void mark() { (void)mcycle(); }

// Turns on matrix state and fills the exp and GeLU tables.
E2E_INLINE void init_tables() {
    asm volatile("csrs mstatus, %0" ::"r"(1U << 29)); // matrix state on
    uint32_t decay = 255U << 16;                      // exp(-i/16) * 255
    for (uint32_t i = 0; i < 256; ++i) {
        dtcm.exp_table[i] = static_cast<uint8_t>(decay >> 16);
        decay = (decay >> 16) * 61565U; // 65536 * exp(-1/16)
        const auto value = static_cast<int8_t>(i);
        dtcm.gelu_table[i] =
            value > 0 ? value : static_cast<int8_t>(value >> 3);
    }
}

// ---- Kernels (one file each) ----------------------------------------------
// embed.cc: token + position embeddings of `tokens` ids into dtcm.residual.
void embed(const uint32_t* ids, uint32_t tokens, uint32_t start_pos);
// layernorm.cc: dtcm.residual -> dtcm.normed, gamma/beta from flash at params.
void layernorm(uint32_t params);
// linear.cc: out = act[in_dim][kBlockTokens] x W[in_dim x out_dim] from flash.
void linear(uint32_t weights, uint32_t in_dim, uint32_t out_dim,
            const int8_t* act, uint32_t tokens, Mode mode, int8_t* out);
// attention.cc: one layer's attention, Q/K/V staged in dtcm.shared.attn,
// output into dtcm.normed; reads and extends that layer's KV cache.
void attention(uint32_t layer, uint32_t tokens, uint32_t start_pos);
// lm_head.cc: tied lm_head on column token_col of dtcm.normed; argmax token.
uint32_t lm_head(uint32_t token_col);

} // namespace npu
