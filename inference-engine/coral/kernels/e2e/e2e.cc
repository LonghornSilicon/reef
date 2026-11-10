// End-to-end GPT-Neo firmware for the Reef NPU (CoralNPU M3 core + Zvt
// matrix engine + DMA), for cycle counts on reef-perf.
//
// Boot: the host loads this program into the 8 KB ITCM over SPI and releases
// the core (reef-perf host_boot). The weights are already in XSPI flash
// (npu.hpp has the map). main() serves the prompt in the mailbox:
//
//   prefill  the prompt in blocks of 16 tokens. Every projection runs on the
//            matrix engine with the 16 tokens as tile rows, so each weight
//            byte streamed from flash is used 16 times.
//   decode   one token per step, same kernels with one tile row.
//
// Everything streams through the 32 KB DTCM by DMA, double buffered, while
// the core computes on the other buffer:
//   linear     weight column blocks (offline-transposed into vtmm B rows)
//   attention  block flash attention: per head, KV blocks of 16 positions,
//              online softmax with a 256-entry exp table (no exp per score),
//              QK^T and PV both on the matrix engine. K is cached transposed
//              (K^T blocks) so it is the contiguous B operand; the newest
//              block is read, the new tokens inserted, and written back.
//
// Activations are feature-major, [feature][kBlockTokens tokens] (kBlockTokens =
// 16, or 8 / 4 for the larger models so everything fits the DTCM): a row is one
// vtmm A operand, and LayerNorm, softmax and requantization vectorize over
// tokens.
//
// Cycle counts only: reef-perf's DMA moves no data and its matrix engine
// computes no tile values, so the generated tokens are meaningless. Every
// loop bound depends only on positions, never on data, so the timing is
// that of real weights.

#include "npu.hpp"

using namespace npu;

#ifndef E2E_PROMPT
#define E2E_PROMPT 32
#endif
#ifndef E2E_NEW_TOKENS
#define E2E_NEW_TOKENS 8
#endif
// E2E_START_POS: decode at this position with no prefill, as if a prompt of
// that length were already in the KV cache (the timing does not depend on what
// the cache holds). 0: prefill the prompt first.
#ifndef E2E_START_POS
#define E2E_START_POS 0
#endif
constexpr uint32_t kMailboxTokens = 64;
static_assert(E2E_START_POS || E2E_PROMPT + E2E_NEW_TOKENS <= kMailboxTokens,
              "mailbox too small");
static_assert((E2E_START_POS ? E2E_START_POS : E2E_PROMPT) + E2E_NEW_TOKENS <=
                  kMaxContext,
              "KV cache too small");
static_assert(E2E_PROMPT + E2E_NEW_TOKENS <= kMaxContext, "KV cache too small");

// The host writes the prompt here before releasing the core and reads the
// generated tokens back. cycles[] holds mcycle at each phase boundary.
struct Mailbox {
    uint32_t prompt_length;
    uint32_t new_tokens;
    uint32_t status;
    uint32_t tokens[kMailboxTokens];
};
__attribute__((section(".mailbox"), used)) Mailbox mailbox = {
    E2E_START_POS ? E2E_START_POS : E2E_PROMPT, E2E_NEW_TOKENS, 0, {}};

namespace {

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
Dtcm dtcm;

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

// ---- Linear: out = act[K][16] x W[K x N] on the matrix engine ------------
void issue_weights(uint32_t weights, uint32_t in_dim, uint32_t chunks,
                   uint32_t index) {
    const uint32_t group = index / chunks;
    const uint32_t chunk = index % chunks;
    const uint32_t rows = (in_dim - chunk * kChunkRows) < kChunkRows
                              ? in_dim - chunk * kChunkRows
                              : kChunkRows;
    dma(weights + group * (in_dim * 64 + 256) + chunk * kChunkRows * 64,
        addr(dtcm.scratch.weights[index & 1]),
        rows * 64 + (chunk == chunks - 1 ? 256 : 0));
}

// Column block col_block, held in the accumulator tile_sel selects.
void epilogue(Mode mode, uint32_t col_block, uint32_t tile_sel,
              const int32_t* bias, int8_t* out, uint32_t tokens) {
    if (mode == kArgmax) { // logits of 16 vocabulary entries, row 0
        const vint32m4_t logits = vtmv(tile_sel);
        const vint32m4_t best = __riscv_vle32_v_i32m4(dtcm.best_logit, 16);
        const vbool8_t greater = __riscv_vmsgt_vv_i32m4_b8(logits, best, 16);
        __riscv_vse32_v_i32m4(
            dtcm.best_logit,
            __riscv_vmerge_vvm_i32m4(best, logits, greater, 16), 16);
        const vint32m4_t index = __riscv_vreinterpret_v_u32m4_i32m4(
            __riscv_vadd_vx_u32m4(__riscv_vid_v_u32m4(16), col_block * 16, 16));
        __riscv_vse32_v_i32m4(
            dtcm.best_index,
            __riscv_vmerge_vvm_i32m4(__riscv_vle32_v_i32m4(dtcm.best_index, 16),
                                     index, greater, 16),
            16);
        return;
    }
    if (mode == kValueRows) { // V rows: token row, 16 dims of head col_block
        const vint32m4_t bias_vec = __riscv_vle32_v_i32m4(bias, 16);
        for (uint32_t row = 0; row < tokens; ++row) {
            vint32m4_t value =
                __riscv_vadd_vv_i32m4(vtmv(tile_sel | row), bias_vec, 16);
            value = __riscv_vsub_vx_i32m4(value, dtcm.act_sum[row], 16);
            __riscv_vse8_v_i8m1(
                out + row * kHidden + col_block * 16,
                to_i8(__riscv_vsra_vx_i32m4(value, kRequantShift, 16)), 16);
        }
        return;
    }
    const vint32m4_t act_sum = __riscv_vle32_v_i32m4(dtcm.act_sum, 16);
    for (uint32_t j = 0; j < 16; ++j) { // feature col_block*16+j over tokens
        int8_t* out_row = out + (col_block * 16 + j) * kBlockTokens;
        vint32m4_t value = __riscv_vsub_vv_i32m4(
            vtmv(tile_sel | kColumnSlice | j), act_sum, 16);
        value = __riscv_vsra_vx_i32m4(__riscv_vadd_vx_i32m4(value, bias[j], 16),
                                      kRequantShift, 16);
        if (mode == kResidual) {
            value = __riscv_vadd_vv_i32m4(value, load_i8(out_row), 16);
        }
        vint8m1_t result = to_i8(value);
        if (mode == kGelu) {
            result = __riscv_vluxei8_v_i8m1(
                dtcm.gelu_table, __riscv_vreinterpret_v_i8m1_u8m1(result), 16);
        }
        __riscv_vse8_v_i8m1(out_row, result, kBlockTokens);
    }
}

void linear(uint32_t weights, uint32_t in_dim, uint32_t out_dim,
            const int8_t* act, uint32_t tokens, Mode mode, int8_t* out) {
    const uint32_t chunks = (in_dim + kChunkRows - 1) / kChunkRows;
    const uint32_t total = out_dim / 64 * chunks;
    issue_weights(weights, in_dim, chunks, 0);
    // Weights are uint8 with zero point 128: subtract 128 * sum_k act[in_dim].
    vint32m4_t act_sum = splat(0);
    for (uint32_t i = 0; i < in_dim; ++i) {
        act_sum =
            __riscv_vadd_vv_i32m4(act_sum, load_i8(act + i * kBlockTokens), 16);
    }
    __riscv_vse32_v_i32m4(dtcm.act_sum, __riscv_vsll_vx_i32m4(act_sum, 7, 16),
                          16);
    for (uint32_t i = 0; i < total; ++i) {
        dma_wait();
        if (i + 1 < total) {
            issue_weights(weights, in_dim, chunks, i + 1);
        }
        const uint32_t chunk = i % chunks;
        const uint32_t rows = (in_dim - chunk * kChunkRows) < kChunkRows
                                  ? in_dim - chunk * kChunkRows
                                  : kChunkRows;
        const uint8_t* buf = dtcm.scratch.weights[i & 1];
        if (chunk == 0) {
            vtzero(true);
        }
        config_int8(tokens);
        mac4(act + chunk * kChunkRows * kBlockTokens, kBlockTokens, buf,
             rows / 4);
        if (chunk == chunks - 1) {
            const auto* bias =
                reinterpret_cast<const int32_t*>(buf + rows * 64);
            for (uint32_t acc = 0; acc < 4; ++acc) {
                epilogue(mode, i / chunks * 4 + acc, acc_tile(acc),
                         bias + acc * 16, out, tokens);
            }
        }
    }
}

// ---- LayerNorm over features, vectorized over tokens: x -> nrm -----------
void layernorm(uint32_t params) {
    dma(params, addr(dtcm.shared.norm_params), 2 * kHidden);
    vint32m4_t sum = splat(0);
    vint32m4_t sum_sq = splat(0);
    for (uint32_t feature = 0; feature < kHidden; ++feature) {
        const vint32m4_t value =
            load_i8(dtcm.residual + feature * kBlockTokens);
        sum = __riscv_vadd_vv_i32m4(sum, value, 16);
        sum_sq = __riscv_vmacc_vv_i32m4(sum_sq, value, value, 16);
    }
    const vint32m4_t mean = __riscv_vdiv_vx_i32m4(sum, kHidden, 16);
    const vint32m4_t var = __riscv_vnmsac_vv_i32m4(
        __riscv_vdiv_vx_i32m4(sum_sq, kHidden, 16), mean, mean, 16);
    const vfloat32m4_t rstd = __riscv_vfrdiv_vf_f32m4(
        __riscv_vfsqrt_v_f32m4(
            __riscv_vfcvt_f_x_v_f32m4(__riscv_vadd_vx_i32m4(var, 1, 16), 16),
            16),
        32.0F, 16);
    dma_wait();
    for (uint32_t feature = 0; feature < kHidden; ++feature) {
        const vint32m4_t centered = __riscv_vsub_vv_i32m4(
            load_i8(dtcm.residual + feature * kBlockTokens), mean, 16);
        vfloat32m4_t out = __riscv_vfmul_vv_f32m4(
            __riscv_vfcvt_f_x_v_f32m4(centered, 16), rstd, 16);
        out = __riscv_vfmul_vf_f32m4(
            out,
            static_cast<float>(dtcm.shared.norm_params[feature]) * 0.03125F,
            16);
        out = __riscv_vfadd_vf_f32m4(
            out, static_cast<float>(dtcm.shared.norm_params[kHidden + feature]),
            16);
        __riscv_vse8_v_i8m1(dtcm.normed + feature * kBlockTokens,
                            to_i8(__riscv_vfcvt_x_f_v_i32m4(out, 16)),
                            kBlockTokens);
    }
}

// ---- Attention ------------------------------------------------------------
// One KV block for a single query token (decode). The score tile has one real
// row, so one row read holds all 16 keys' scores: the softmax runs on that
// one vector (reductions across keys), and P V comes back as rows too.
// State uses lane 0 of row_max / row_sum; out_acc[dim] holds the output.
void attend_row(uint32_t head, uint32_t block, const uint8_t* key_block,
                const uint8_t* value_block, uint32_t query_pos, bool local) {
    Softmax& state = dtcm.scratch.softmax;
    vtzero();
    config_int8(1);
    mac<true>(dtcm.shared.attn.query + head * kHeadDim * kBlockTokens,
              kBlockTokens, key_block, 16, kHeadDim / 4); // S = q K^T
    // Key positions of this block's 16 lanes; visible: oldest <= key <= query.
    const vuint32m4_t key_pos =
        __riscv_vadd_vx_u32m4(__riscv_vid_v_u32m4(16), block * 16, 16);
    const uint32_t oldest =
        (local && query_pos >= kLocalWindow) ? query_pos - kLocalWindow + 1 : 0;
    const vbool8_t valid = __riscv_vmand_mm_b8(
        __riscv_vmsleu_vx_u32m4_b8(key_pos, query_pos, 16),
        __riscv_vmsgeu_vx_u32m4_b8(key_pos, oldest, 16), 16);
    const vint32m4_t scores =
        __riscv_vmerge_vvm_i32m4(splat(kNegInf), vtmv(0), valid, 16);
    const int32_t block_max =
        __riscv_vmv_x_s_i32m1_i32(__riscv_vredmax_vs_i32m4_i32m1(
            scores, __riscv_vmv_s_x_i32m1(kNegInf, 1), 16));
    const int32_t m_old = state.row_max[0];
    const int32_t m_new = block_max > m_old ? block_max : m_old;
    const uint32_t gap = static_cast<uint32_t>(m_new - m_old) >> kExpShift;
    const int32_t alpha = dtcm.exp_table[gap < 255 ? gap : 255];
    vint32m4_t prob = exp_lut(__riscv_vrsub_vx_i32m4(scores, m_new, 16));
    prob = __riscv_vmerge_vvm_i32m4(splat(0), prob, valid, 16);
    const int32_t prob_sum = __riscv_vmv_x_s_i32m1_i32(
        __riscv_vredsum_vs_i32m4_i32m1(prob, __riscv_vmv_s_x_i32m1(0, 1), 16));
    // P as 16 contiguous bytes; the A operand reads it with stride 1, so row t
    // starts at probs + t and its lane 0 is key t's probability.
    __riscv_vse8_v_u8m1(state.probs,
                        __riscv_vreinterpret_v_i8m1_u8m1(
                            to_i8(__riscv_vsub_vx_i32m4(prob, 128, 16))),
                        16);
    state.row_max[0] = m_new;
    state.row_sum[0] = ((state.row_sum[0] * alpha) >> 8) + prob_sum;
    for (uint32_t dim0 = 0; dim0 < kHeadDim; dim0 += 16) { // O += P V
        vtzero();
        config_int8(1);
        mac<false>(state.probs, 1, value_block + dim0, kHeadDim, 4);
        int32_t* acc = state.out_acc + dim0;
        const vint32m4_t scaled = __riscv_vsra_vx_i32m4(
            __riscv_vmul_vx_i32m4(__riscv_vle32_v_i32m4(acc, 16), alpha, 16), 8,
            16);
        __riscv_vse32_v_i32m4(acc, __riscv_vadd_vv_i32m4(scaled, vtmv(0), 16),
                              16);
    }
}

// One KV block (16 positions) for head h: online-softmax update of m, l, acc.
void attend_block(uint32_t head, uint32_t block, const uint8_t* key_block,
                  const uint8_t* value_block, uint32_t tokens,
                  uint32_t start_pos, bool local) {
    if (tokens == 1) {
        attend_row(head, block, key_block, value_block, start_pos, local);
        return;
    }
    Softmax& state = dtcm.scratch.softmax;
    vtzero();
    config_int8(tokens);
    mac<true>(dtcm.shared.attn.query + head * kHeadDim * kBlockTokens,
              kBlockTokens, key_block, 16, kHeadDim / 4); // S = Q K^T
    const vuint32m4_t query_pos =
        __riscv_vadd_vx_u32m4(__riscv_vid_v_u32m4(16), start_pos, 16);
    const vint32m4_t neg_inf = splat(kNegInf);
    auto valid = [&](uint32_t key) {
        vbool8_t ok = __riscv_vmsgeu_vx_u32m4_b8(query_pos, key, 16);
        if (local) {
            ok = __riscv_vmand_mm_b8(
                ok,
                __riscv_vmsltu_vx_u32m4_b8(query_pos, key + kLocalWindow, 16),
                16);
        }
        return ok;
    };
    vint32m4_t block_max = neg_inf;
    for (uint32_t j = 0; j < 16; ++j) {
        block_max = __riscv_vmax_vv_i32m4(
            block_max,
            __riscv_vmerge_vvm_i32m4(neg_inf, vtmv(kColumnSlice | j),
                                     valid(block * 16 + j), 16),
            16);
    }
    const vint32m4_t m_old = __riscv_vle32_v_i32m4(state.row_max, 16);
    const vint32m4_t m_new = __riscv_vmax_vv_i32m4(m_old, block_max, 16);
    const vint32m4_t alpha = exp_lut(__riscv_vsub_vv_i32m4(m_new, m_old, 16));
    vint32m4_t prob_sum = splat(0);
    for (uint32_t j = 0; j < 16; ++j) {
        vint32m4_t prob =
            exp_lut(__riscv_vsub_vv_i32m4(m_new, vtmv(kColumnSlice | j), 16));
        prob =
            __riscv_vmerge_vvm_i32m4(splat(0), prob, valid(block * 16 + j), 16);
        prob_sum = __riscv_vadd_vv_i32m4(prob_sum, prob, 16);
        __riscv_vse8_v_u8m1(state.probs + j * 16,
                            __riscv_vreinterpret_v_i8m1_u8m1(
                                to_i8(__riscv_vsub_vx_i32m4(prob, 128, 16))),
                            16);
    }
    __riscv_vse32_v_i32m4(state.row_max, m_new, 16);
    __riscv_vse32_v_i32m4(
        state.row_sum,
        __riscv_vadd_vv_i32m4(
            __riscv_vsra_vx_i32m4(
                __riscv_vmul_vv_i32m4(__riscv_vle32_v_i32m4(state.row_sum, 16),
                                      alpha, 16),
                8, 16),
            prob_sum, 16),
        16);
    for (uint32_t dim0 = 0; dim0 < kHeadDim;
         dim0 += 16) { // O += P V, 16 dims per tile
        vtzero();
        config_int8(tokens);
        mac<false>(state.probs, 16, value_block + dim0, kHeadDim, 4);
        for (uint32_t j = dim0; j < kHeadDim && j < dim0 + 16; ++j) {
            int32_t* acc_row = state.out_acc + j * 16;
            vint32m4_t value = __riscv_vmul_vv_i32m4(
                __riscv_vle32_v_i32m4(acc_row, 16), alpha, 16);
            value = __riscv_vadd_vv_i32m4(__riscv_vsra_vx_i32m4(value, 8, 16),
                                          vtmv(kColumnSlice | (j - dim0)), 16);
            __riscv_vse32_v_i32m4(acc_row, value, 16);
        }
    }
}

void attention(uint32_t layer, uint32_t tokens, uint32_t start_pos) {
    const bool local = (layer & 1) != 0;
    const uint32_t tail_col = start_pos & 15;
    const uint32_t tail_block = start_pos >> 4;
    const uint32_t first_block = (local && start_pos + 1 > kLocalWindow)
                                     ? (start_pos + 1 - kLocalWindow) >> 4
                                     : 0;
    const uint32_t key_base =
        kKeyCache + layer * kNumHeads * kKvBlocks * kKvBlock;
    const uint32_t value_base =
        kValueCache + layer * kNumHeads * kKvBlocks * kKvBlock;
    auto issue = [&](uint32_t head, uint32_t start_block, uint32_t buf) {
        const uint32_t count = (tail_block + 1 - start_block) < kKvChunk
                                   ? tail_block + 1 - start_block
                                   : kKvChunk;
        const uint32_t offset = (head * kKvBlocks + start_block) * kKvBlock;
        dma(key_base + offset, addr(dtcm.shared.attn.kv_buf[buf][0]),
            count * kKvBlock);
        dma(value_base + offset, addr(dtcm.shared.attn.kv_buf[buf][1]),
            count * kKvBlock);
    };
    uint32_t head = 0;
    uint32_t start_block = first_block;
    uint32_t buf = 0;
    issue(head, start_block, buf);
    while (true) {
        dma_wait();
        uint32_t next_head = head;
        uint32_t next_start = start_block + kKvChunk;
        if (next_start > tail_block) {
            ++next_head;
            next_start = first_block;
        }
        if (next_head < kNumHeads) {
            issue(next_head, next_start, buf ^ 1);
        }
        Softmax& state = dtcm.scratch.softmax;
        if (start_block == first_block) {
            __riscv_vse32_v_i32m4(state.row_max, splat(kNegInf), 16);
            __riscv_vse32_v_i32m4(state.row_sum, splat(0), 16);
            for (uint32_t j = 0; j < kHeadDim; ++j) {
                __riscv_vse32_v_i32m4(state.out_acc + j * 16, splat(0), 16);
            }
        }
        for (uint32_t block = start_block;
             block <= tail_block && block < start_block + kKvChunk; ++block) {
            uint8_t* key_block = dtcm.shared.attn.kv_buf[buf][0] +
                                 (block - start_block) * kKvBlock;
            uint8_t* value_block = dtcm.shared.attn.kv_buf[buf][1] +
                                   (block - start_block) * kKvBlock;
            if (block ==
                tail_block) { // insert the new tokens at positions tail_col..
                for (uint32_t j = 0; j < kHeadDim; ++j) {
                    __riscv_vse8_v_i8m1(
                        reinterpret_cast<int8_t*>(key_block) + j * 16 +
                            tail_col,
                        __riscv_vle8_v_i8m1(dtcm.shared.attn.key +
                                                (head * kHeadDim + j) *
                                                    kBlockTokens,
                                            tokens),
                        tokens);
                }
                for (uint32_t row = 0; row < tokens; ++row) {
                    __riscv_vse8_v_u8m4(
                        value_block + (tail_col + row) * kHeadDim,
                        __riscv_vle8_v_u8m4(dtcm.shared.attn.value +
                                                row * kHidden + head * kHeadDim,
                                            kHeadDim),
                        kHeadDim);
                }
            }
            attend_block(head, block, key_block, value_block, tokens, start_pos,
                         local);
            if (block == tail_block) { // write the newest block back
                const uint32_t offset = (head * kKvBlocks + block) * kKvBlock;
                dma(addr(key_block), key_base + offset, kKvBlock);
                dma(addr(value_block), value_base + offset, kKvBlock);
            }
        }
        if (start_block + kKvChunk >
            tail_block) {      // head done: O = (acc - 128 row_sum) / row_sum
            if (tokens == 1) { // one token: out_acc[dim], scalar row_sum
                const int32_t row_sum = state.row_sum[0];
                const int32_t recip = static_cast<int32_t>(
                    (1U << 20) / static_cast<uint32_t>(row_sum));
                for (uint32_t dim0 = 0; dim0 < kHeadDim; dim0 += 16) {
                    vint32m4_t out = __riscv_vsub_vx_i32m4(
                        __riscv_vle32_v_i32m4(state.out_acc + dim0, 16),
                        row_sum << 7, 16);
                    out = __riscv_vsra_vx_i32m4(
                        __riscv_vmul_vx_i32m4(out, recip, 16), 20, 16);
                    // Dims down column 0 of normed (rows kBlockTokens apart).
                    __riscv_vsse8_v_i8m1(
                        dtcm.normed + (head * kHeadDim + dim0) * kBlockTokens,
                        kBlockTokens, to_i8(out),
                        kHeadDim - dim0 < 16 ? kHeadDim - dim0 : 16);
                }
            } else { // block of tokens: out_acc[dim][token], row_sum per token
                const vint32m4_t row_sum =
                    __riscv_vle32_v_i32m4(state.row_sum, 16);
                const vint32m4_t recip =
                    __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vdivu_vv_u32m4(
                        __riscv_vmv_v_x_u32m4(1U << 20, 16),
                        __riscv_vreinterpret_v_i32m4_u32m4(row_sum), 16));
                const vint32m4_t zero_point =
                    __riscv_vsll_vx_i32m4(row_sum, 7, 16);
                for (uint32_t j = 0; j < kHeadDim; ++j) {
                    vint32m4_t out_row = __riscv_vsub_vv_i32m4(
                        __riscv_vle32_v_i32m4(state.out_acc + j * 16, 16),
                        zero_point, 16);
                    out_row = __riscv_vsra_vx_i32m4(
                        __riscv_vmul_vv_i32m4(out_row, recip, 16), 20, 16);
                    __riscv_vse8_v_i8m1(dtcm.normed + (head * kHeadDim + j) *
                                                          kBlockTokens,
                                        to_i8(out_row), kBlockTokens);
                }
            }
        }
        if (next_head >= kNumHeads) {
            break;
        }
        head = next_head;
        start_block = next_start;
        buf ^= 1;
    }
    dma_wait(); // the last write-backs
}

// ---- Model ----------------------------------------------------------------
void embed(const uint32_t* ids, uint32_t tokens, uint32_t start_pos) {
    for (uint32_t j = 0; j < tokens; ++j) {
        const uint32_t id = ids[j] < kVocab ? ids[j] : 0;
        dma(kTokenEmbed + id * kHidden, addr(dtcm.shared.embed_rows[j][0]),
            kHidden);
        dma(kPosEmbed + (start_pos + j) * kHidden,
            addr(dtcm.shared.embed_rows[j][1]), kHidden);
    }
    dma_wait();
    for (uint32_t j = 0; j < tokens; ++j) {
        for (uint32_t feature = 0; feature < kHidden; feature += 16) {
            __riscv_vsse8_v_i8m1(
                dtcm.residual + feature * kBlockTokens + j, kBlockTokens,
                __riscv_vsadd_vv_i8m1(
                    __riscv_vle8_v_i8m1(dtcm.shared.embed_rows[j][0] + feature,
                                        16),
                    __riscv_vle8_v_i8m1(dtcm.shared.embed_rows[j][1] + feature,
                                        16),
                    16),
                16);
        }
    }
}

// mcycle read: reef-perf lists every read's cycle (JSON mcycle_reads), so a
// mark after each kernel times it. 7 per layer, 1 per embed, 2 per head.
E2E_INLINE void mark() { (void)mcycle(); }

void layers(uint32_t tokens, uint32_t start_pos) {
    for (uint32_t layer = 0; layer < kNumLayers; ++layer) {
        const uint32_t base = kLayersBase + layer * kLayerBytes;
        layernorm(base + kNormOffset);
        mark(); // ln1
        linear(base + kQueryOffset, kHidden, kHidden, dtcm.normed, tokens,
               kFeatureMajor, dtcm.shared.attn.query);
        linear(base + kKeyOffset, kHidden, kHidden, dtcm.normed, tokens,
               kFeatureMajor, dtcm.shared.attn.key);
        linear(base + kValueOffset, kHidden, kHidden, dtcm.normed, tokens,
               kValueRows, reinterpret_cast<int8_t*>(dtcm.shared.attn.value));
        mark(); // qkv
        attention(layer, tokens, start_pos);
        mark(); // attention
        linear(base + kOutOffset, kHidden, kHidden, dtcm.normed, tokens,
               kResidual, dtcm.residual);
        mark(); // attn_out + residual
        layernorm(base + kNormOffset + 2 * kHidden);
        mark(); // ln2
        linear(base + kFc1Offset, kHidden, kMlpDim, dtcm.normed, tokens, kGelu,
               dtcm.shared.mlp);
        mark(); // fc1 + gelu
        linear(base + kFc2Offset, kMlpDim, kHidden, dtcm.shared.mlp, tokens,
               kResidual, dtcm.residual);
        mark(); // fc2 + residual
    }
}

// ln_f, then the tied lm_head for token column `last`; greedy argmax.
uint32_t next_token(uint32_t token_col) {
    layernorm(kFinalNorm);
    mark(); // ln_f
    __riscv_vse32_v_i32m4(dtcm.best_logit, splat(kNegInf), 16);
    linear(kLmHead, kHidden, kVocabPadded, dtcm.normed + token_col, 1, kArgmax,
           nullptr);
    const vint32m4_t best = __riscv_vle32_v_i32m4(dtcm.best_logit, 16);
    const int32_t max_logit =
        __riscv_vmv_x_s_i32m1_i32(__riscv_vredmax_vs_i32m4_i32m1(
            best, __riscv_vmv_s_x_i32m1(kNegInf, 1), 16));
    const long lane =
        __riscv_vfirst_m_b8(__riscv_vmseq_vx_i32m4_b8(best, max_logit, 16), 16);
    mark(); // lm_head
    return static_cast<uint32_t>(dtcm.best_index[lane < 0 ? 0 : lane]);
}

} // namespace

int main() {
    asm volatile("csrs mstatus, %0" ::"r"(1U << 29)); // matrix state on
    uint32_t decay = 255U << 16;                      // exp(-i/16) * 255
    for (uint32_t i = 0; i < 256; ++i) {
        dtcm.exp_table[i] = static_cast<uint8_t>(decay >> 16);
        decay = (decay >> 16) * 61565U; // 65536 * exp(-1/16)
        const auto value = static_cast<int8_t>(i);
        dtcm.gelu_table[i] =
            value > 0 ? value : static_cast<int8_t>(value >> 3);
    }
    Mailbox& box = mailbox;
    mark(); // ROI start
    const uint32_t prompt = box.prompt_length;
    uint32_t next = 0;
    if (!E2E_START_POS) {
        uint32_t tokens = kBlockTokens;
        for (uint32_t start_pos = 0; start_pos < prompt;
             start_pos += kBlockTokens) {
            tokens = prompt - start_pos < kBlockTokens ? prompt - start_pos
                                                       : kBlockTokens;
            embed(box.tokens + start_pos, tokens, start_pos);
            mark(); // embedding
            layers(tokens, start_pos);
        }
        next = next_token(tokens - 1);
    }
    for (uint32_t step = 0; step < box.new_tokens; ++step) {
        const uint32_t pos = prompt + step;
        box.tokens[(E2E_START_POS ? step : pos) % kMailboxTokens] = next;
        if (E2E_START_POS || step + 1 < box.new_tokens) {
            embed(&next, 1, pos);
            mark(); // embedding
            layers(1, pos);
            next = next_token(0);
        }
    }
    box.status = 1;
    return 0;
}
