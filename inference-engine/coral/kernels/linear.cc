// Linear layer on the Zvt matrix unit, weights streamed from flash by DMA.

#include "common.hpp"

namespace npu {

// ---- Linear: out = act[K][16] x W[K x N] on the matrix engine ------------
static void issue_weights(uint32_t weights, uint32_t in_dim, uint32_t chunks,
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
static void epilogue(Mode mode, uint32_t col_block, uint32_t tile_sel,
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

} // namespace npu
