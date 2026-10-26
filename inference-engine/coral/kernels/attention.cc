// One attention head over cached keys and values (Attention::context).
//
// The reference computes scores and probability products for every key and
// gives masked keys a probability of exactly zero. A zero product added to
// a running sum leaves it unchanged, so visiting only the visible keys, in
// the same order, gives the same bits. The int8 head is integer throughout.

#include "fixed_point.hpp"
#include "reef_kernels/kernels.hpp"
#include "vector_math.hpp"

#include <riscv_vector.h>

namespace reef_kernels {

void attention_head_f32(std::size_t query_len, std::size_t key_len,
                        std::size_t head_dim, std::size_t query_start,
                        std::size_t window, const float* q,
                        std::size_t q_stride, const float* k, const float* v,
                        float* scores, float* out, std::size_t out_stride) {
    const std::ptrdiff_t row_bytes =
        static_cast<std::ptrdiff_t>(head_dim * sizeof(float));
    for (std::size_t i = 0; i < query_len; ++i) {
        // Visible keys [first, end): causal, then the sliding window.
        const std::size_t position = query_start + i;
        const std::size_t end = position + 1 < key_len ? position + 1 : key_len;
        const std::size_t first =
            window > 0 && position + 1 > window ? position + 1 - window : 0;
        const float* qi = q + (i * q_stride);

        // scores[j] = sum over d, in order, of q[d] * k[j][d]; the lanes walk
        // keys, reading one column of k with a strided load.
        for (std::size_t j = first; j < end;) {
            const std::size_t vl = __riscv_vsetvl_e32m4(end - j);
            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0F, vl);
            for (std::size_t d = 0; d < head_dim; ++d) {
                const vfloat32m4_t column = __riscv_vlse32_v_f32m4(
                    k + (j * head_dim) + d, row_bytes, vl);
                acc = __riscv_vfadd_vv_f32m4(
                    acc, __riscv_vfmul_vf_f32m4(column, qi[d], vl), vl);
            }
            __riscv_vse32_v_f32m4(scores + j, acc, vl);
            j += vl;
        }

        // masked_softmax: row maximum, exp and ordered sum, then divide.
        vfloat32m1_t largest = __riscv_vfmv_s_f_f32m1(scores[first], 1);
        for (std::size_t j = first; j < end;) {
            const std::size_t vl = __riscv_vsetvl_e32m4(end - j);
            largest = __riscv_vfredmax_vs_f32m4_f32m1(
                __riscv_vle32_v_f32m4(scores + j, vl), largest, vl);
            j += vl;
        }
        const float row_max = __riscv_vfmv_f_s_f32m1_f32(largest);
        vfloat32m1_t sum = __riscv_vfmv_s_f_f32m1(0.0F, 1);
        for (std::size_t j = first; j < end;) {
            const std::size_t vl = __riscv_vsetvl_e32m4(end - j);
            const vfloat32m4_t e = detail::exp_nonpositive(
                __riscv_vfsub_vf_f32m4(__riscv_vle32_v_f32m4(scores + j, vl),
                                       row_max, vl),
                vl);
            __riscv_vse32_v_f32m4(scores + j, e, vl);
            sum = __riscv_vfredosum_vs_f32m4_f32m1(e, sum, vl);
            j += vl;
        }
        const float total = __riscv_vfmv_f_s_f32m1_f32(sum);
        for (std::size_t j = first; j < end;) {
            const std::size_t vl = __riscv_vsetvl_e32m4(end - j);
            __riscv_vse32_v_f32m4(
                scores + j,
                __riscv_vfdiv_vf_f32m4(__riscv_vle32_v_f32m4(scores + j, vl),
                                       total, vl),
                vl);
            j += vl;
        }

        // out[c] = sum over visible j, in order, of p[j] * v[j][c].
        float* oi = out + (i * out_stride);
        for (std::size_t c = 0; c < head_dim;) {
            const std::size_t vl = __riscv_vsetvl_e32m4(head_dim - c);
            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0F, vl);
            for (std::size_t j = first; j < end; ++j) {
                const vfloat32m4_t row =
                    __riscv_vle32_v_f32m4(v + (j * head_dim) + c, vl);
                acc = __riscv_vfadd_vv_f32m4(
                    acc, __riscv_vfmul_vf_f32m4(row, scores[j], vl), vl);
            }
            __riscv_vse32_v_f32m4(oi + c, acc, vl);
            c += vl;
        }
    }
}

void attention_head_int8(std::size_t query_len, std::size_t key_len,
                         std::size_t head_dim, std::size_t query_start,
                         std::size_t window, const std::int8_t* q,
                         std::size_t q_stride, const std::int8_t* k,
                         const std::int8_t* v, unsigned score_shift,
                         const std::uint16_t* table, unsigned out_shift,
                         std::int32_t* scores, std::uint8_t* probs,
                         std::int8_t* out, std::size_t out_stride) {
    const auto row_bytes = static_cast<std::ptrdiff_t>(head_dim);
    for (std::size_t i = 0; i < query_len; ++i) {
        const std::size_t position = query_start + i;
        const std::size_t end = position + 1 < key_len ? position + 1 : key_len;
        const std::size_t first =
            window > 0 && position + 1 > window ? position + 1 - window : 0;
        const std::int8_t* qi = q + (i * q_stride);

        // scores[j] = q . k[j] in int32; the lanes walk keys.
        for (std::size_t j = first; j < end;) {
            const std::size_t vl = __riscv_vsetvl_e32m4(end - j);
            vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
            for (std::size_t d = 0; d < head_dim; ++d) {
                const vint16m2_t column = __riscv_vsext_vf2_i16m2(
                    __riscv_vlse8_v_i8m1(k + (j * head_dim) + d, row_bytes, vl),
                    vl);
                acc = __riscv_vwmacc_vx_i32m4(acc, qi[d], column, vl);
            }
            __riscv_vse32_v_i32m4(scores + j, acc, vl);
            j += vl;
        }

        // Softmax over the visible keys only, as masked_softmax_int8.
        softmax_int8(end - first, scores + first, score_shift, table,
                     probs + first);

        // out[c] = saturate(rounding_shift(sum over visible j of p[j] *
        // v[j][c], out_shift)).
        std::int8_t* oi = out + (i * out_stride);
        for (std::size_t c = 0; c < head_dim;) {
            const std::size_t vl = __riscv_vsetvl_e8m1(head_dim - c);
            vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
            for (std::size_t j = first; j < end; ++j) {
                const vint16m2_t row = __riscv_vsext_vf2_i16m2(
                    __riscv_vle8_v_i8m1(v + (j * head_dim) + c, vl), vl);
                acc = __riscv_vwmacc_vx_i32m4(
                    acc, static_cast<std::int16_t>(probs[j]), row, vl);
            }
            __riscv_vse8_v_i8m1(oi + c, detail::narrow_int8(acc, out_shift, vl),
                                vl);
            c += vl;
        }
    }
}

} // namespace reef_kernels
