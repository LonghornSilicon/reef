// Block flash attention over the KV cache in flash.

#include "common.hpp"

namespace npu {

// ---- Attention ------------------------------------------------------------
// One KV block for a single query token (decode). The score tile has one real
// row, so one row read holds all 16 keys' scores: the softmax runs on that
// one vector (reductions across keys), and P V comes back as rows too.
// State uses lane 0 of row_max / row_sum; out_acc[dim] holds the output.
static void attend_row(uint32_t head, uint32_t block, const uint8_t* key_block,
                       const uint8_t* value_block, uint32_t query_pos,
                       bool local) {
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
static void attend_block(uint32_t head, uint32_t block,
                         const uint8_t* key_block, const uint8_t* value_block,
                         uint32_t tokens, uint32_t start_pos, bool local) {
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

} // namespace npu
