#pragma once

// Coral M3 kernels: the GPT-Neo operators, vectorized with RVV intrinsics.
//
// Each kernel reproduces one host reference function in
// include/inference_engine/ bit for bit: it performs the same float
// operations in the same order, so it rounds identically. Integer sums may be
// reordered freely. coral/tests/ checks every kernel against golden outputs
// that coral/host/gen_golden.cpp computes with the reference, and fails on
// any differing bit. A kernel change that the reference does not share fails
// those tests; change the reference first, then the kernel.
//
// Kernels only compute: no allocation, no globals, no knowledge of the host
// or of where data lives. Every pointer is valid for the sizes given.
// Matrices are row-major. Linear weights are stored transposed, [in][out],
// so one row of the weight holds a contiguous strip of outputs.

#include <cstddef>
#include <cstdint>

namespace reef_kernels {

// linear() without a bias (operator/linear.hpp):
//   y[r][c] = sum over k, in order, of x[r][k] * wt[k][c]
// x is [rows][in], wt is the weight transposed [in][out], y is [rows][out].
void linear_f32(std::size_t rows, std::size_t in, std::size_t out,
                const float* x, const float* wt, float* y);

// linear() with a bias: linear_f32, then y[r][c] + bias[c].
void linear_bias_f32(std::size_t rows, std::size_t in, std::size_t out,
                     const float* x, const float* wt, const float* bias,
                     float* y);

// The W8A8 linear() overload (operator/linear.hpp, quant.hpp), no bias.
// Each row of x is quantized with its own scale into x_q (scratch, [in]),
// multiplied by the int8 weight wt ([in][out], transposed) in int32, and
// rescaled as float(sum) * (row_scale * w_scale[c]).
void linear_w8a8(std::size_t rows, std::size_t in, std::size_t out,
                 const float* x, const std::int8_t* wt, const float* w_scale,
                 std::int8_t* x_q, float* y);

// The W8A8 linear() overload with a bias added last.
void linear_w8a8_bias(std::size_t rows, std::size_t in, std::size_t out,
                      const float* x, const std::int8_t* wt,
                      const float* w_scale, const float* bias, std::int8_t* x_q,
                      float* y);

// linear(x, w) for one row with w in its natural [out][in] layout:
//   y[c] = sum over k, in order, of x[k] * w[c][k]
// GPT-Neo's lm_head reuses the token embedding table this way.
void matvec_rows_f32(std::size_t out, std::size_t in, const float* w,
                     const float* x, float* y);

// layernorm() (operator/layernorm.hpp) over rows of width values.
void layernorm_f32(std::size_t rows, std::size_t width, const float* x,
                   const float* gain, const float* shift, float eps, float* y);

// gelu() for float (operator/activation.hpp), elementwise.
void gelu_f32(std::size_t n, const float* x, float* y);

// exp_nonpositive() and tanh_float() (math.hpp), elementwise. The other
// kernels use the same vector code; these entry points let tests check it.
void exp_nonpositive_f32(std::size_t n, const float* x, float* y);
void tanh_f32(std::size_t n, const float* x, float* y);

// Attention::context() (operator/attention.hpp) for one head:
// scores q.k^T, masked_softmax() over the visible keys, then probs.v.
// Query row i sits at position query_start + i and sees keys up to its own
// position; with window > 0 (LocalAttention) only the last window of them.
// window == 0 means global attention.
//   q:      query_len rows of head_dim values, q_stride floats apart
//   k, v:   key_len rows of head_dim values each, contiguous
//   scores: scratch, key_len floats
//   out:    query_len rows of head_dim values, out_stride floats apart
void attention_head_f32(std::size_t query_len, std::size_t key_len,
                        std::size_t head_dim, std::size_t query_start,
                        std::size_t window, const float* q,
                        std::size_t q_stride, const float* k, const float* v,
                        float* scores, float* out, std::size_t out_stride);

// lookup_token_embeddings() then add_position_embeddings() for one token:
//   y[c] = wte[id][c] + wpe[position][c]
void embedding_f32(std::size_t width, std::uint32_t id, std::uint32_t position,
                   const float* wte, const float* wpe, float* y);

// Elementwise a + b, the residual add of a GPT-Neo block.
void add_f32(std::size_t n, const float* a, const float* b, float* y);

// argmax() (model.hpp): index of the first largest value.
std::uint32_t argmax_f32(std::size_t n, const float* x);

// ---------------------------------------------------------------------------
// Integer-only operators (int8.hpp): int8 activations, int32 sums, rounding
// right shifts (vnclip, vxrm = RNU) and host-built lookup tables. Integer
// sums may be reordered freely, so these vectorize without constraints.

// linear_int8() without a bias:
//   y[r][c] = saturate(rounding_shift(sum over k of x[r][k] * wt[k][c],
//                                     shift))
// wt is the int8 weight transposed, [in][out].
void linear_int8(std::size_t rows, std::size_t in, std::size_t out,
                 const std::int8_t* x, const std::int8_t* wt, unsigned shift,
                 std::int8_t* y);

// linear_int8() with an int32 bias added before the shift.
void linear_bias_int8(std::size_t rows, std::size_t in, std::size_t out,
                      const std::int8_t* x, const std::int8_t* wt,
                      const std::int32_t* bias, unsigned shift, std::int8_t* y);

// gelu_int8(): y[i] = table[x[i] + 128], with a table from gelu_table().
void gelu_int8(std::size_t n, const std::int8_t* x, const std::int8_t* table,
               std::int8_t* y);

// masked_softmax_int8() for one row with every key visible: int32 scores
// to uint8 probabilities summing to about 255, through an exp_table().
void softmax_int8(std::size_t n, const std::int32_t* scores, unsigned shift,
                  const std::uint16_t* table, std::uint8_t* probs);

// attend_head_int8() for one head over cached keys and values; the query
// positions and window work as in attention_head_f32.
//   scores: scratch, key_len int32;  probs: scratch, key_len uint8
void attention_head_int8(std::size_t query_len, std::size_t key_len,
                         std::size_t head_dim, std::size_t query_start,
                         std::size_t window, const std::int8_t* q,
                         std::size_t q_stride, const std::int8_t* k,
                         const std::int8_t* v, unsigned score_shift,
                         const std::uint16_t* table, unsigned out_shift,
                         std::int32_t* scores, std::uint8_t* probs,
                         std::int8_t* out, std::size_t out_stride);

// layernorm_int8() over rows of width values; any width.
void layernorm_int8(std::size_t rows, std::size_t width, const std::int8_t* x,
                    const std::int8_t* gain, const std::int8_t* shift,
                    std::int8_t* y);

// add_int8(): saturating a + b, for addends at the same scale.
void add_int8(std::size_t n, const std::int8_t* a, const std::int8_t* b,
              std::int8_t* y);

// One token's embedding in int8: y[c] = saturate(wte[id][c] +
// wpe[position][c]), tables at the same scale.
void embedding_int8(std::size_t width, std::uint32_t id, std::uint32_t position,
                    const std::int8_t* wte, const std::int8_t* wpe,
                    std::int8_t* y);

} // namespace reef_kernels
