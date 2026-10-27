#pragma once

// Shapes of the golden test cases. coral/host/gen_golden.cpp writes one
// header per test into the build tree (golden/<test>.hpp), filling arrays of
// these with inputs and the outputs the host reference computes from them.

#include <cstddef>
#include <cstdint>

namespace golden {

// Elementwise: y = f(x).
struct UnaryCase {
    std::size_t n;
    const float* x;
    const float* y;
};

// linear_f32: wt is the weight transposed, [in][out].
struct LinearCase {
    std::size_t rows, in, out;
    const float* x;
    const float* wt;
    const float* y;
};

// linear_bias_f32.
struct LinearBiasCase {
    std::size_t rows, in, out;
    const float* x;
    const float* wt;
    const float* bias;
    const float* y;
};

// linear_w8a8: wt is the int8 weight transposed, [in][out].
struct W8a8Case {
    std::size_t rows, in, out;
    const float* x;
    const std::int8_t* wt;
    const float* w_scale;
    const float* y;
};

// linear_w8a8_bias.
struct W8a8BiasCase {
    std::size_t rows, in, out;
    const float* x;
    const std::int8_t* wt;
    const float* w_scale;
    const float* bias;
    const float* y;
};

// matvec_rows_f32: w in its natural [out][in] layout.
struct MatvecCase {
    std::size_t out, in;
    const float* w;
    const float* x;
    const float* y;
};

struct LayerNormCase {
    std::size_t rows, width;
    float eps;
    const float* x;
    const float* gain;
    const float* shift;
    const float* y;
};

// attention_head_f32 with q and out packed (stride == head_dim).
struct AttentionCase {
    std::size_t query_len, key_len, head_dim, query_start, window;
    const float* q;
    const float* k;
    const float* v;
    const float* y;
};

struct AddCase {
    std::size_t n;
    const float* a;
    const float* b;
    const float* y;
};

struct EmbeddingCase {
    std::size_t width;
    std::uint32_t id, position;
    const float* wte;
    const float* wpe;
    const float* y;
};

struct ArgmaxCase {
    std::size_t n;
    const float* x;
    std::uint32_t index;
};

// Integer-only operators (int8.hpp).

struct Int8LinearCase {
    std::size_t rows, in, out;
    unsigned shift;
    const std::int8_t* x;
    const std::int8_t* wt; // [in][out]
    const std::int8_t* y;
};

struct Int8LinearBiasCase {
    std::size_t rows, in, out;
    unsigned shift;
    const std::int8_t* x;
    const std::int8_t* wt; // [in][out]
    const std::int32_t* bias;
    const std::int8_t* y;
};

// gelu_int8: y = table[x + 128].
struct Int8TableCase {
    std::size_t n;
    const std::int8_t* table; // 256 entries
    const std::int8_t* x;
    const std::int8_t* y;
};

// softmax_int8, one row per n scores.
struct Int8SoftmaxCase {
    std::size_t rows, n;
    unsigned shift;
    const std::uint16_t* table; // 256 entries
    const std::int32_t* scores;
    const std::uint8_t* probs;
};

// attention_head_int8 with q and out packed (stride == head_dim).
struct Int8AttentionCase {
    std::size_t query_len, key_len, head_dim, query_start, window;
    unsigned score_shift, out_shift;
    const std::uint16_t* table;
    const std::int8_t* q;
    const std::int8_t* k;
    const std::int8_t* v;
    const std::int8_t* y;
};

struct Int8LayerNormCase {
    std::size_t rows, width;
    const std::int8_t* x;
    const std::int8_t* gain;
    const std::int8_t* shift;
    const std::int8_t* y;
};

struct Int8AddCase {
    std::size_t n;
    const std::int8_t* a;
    const std::int8_t* b;
    const std::int8_t* y;
};

struct Int8EmbeddingCase {
    std::size_t width;
    std::uint32_t id, position;
    const std::int8_t* wte;
    const std::int8_t* wpe;
    const std::int8_t* y;
};

// A whole model: a packed blob (model_blob.hpp), a prompt, and what the
// reference's GPTNeo::generate produced from it, with the logits that chose
// each generated token.
struct ModelCase {
    const std::uint32_t* blob; // 16-byte aligned
    std::size_t prompt_length;
    const std::uint32_t* prompt;
    std::size_t new_tokens;
    const std::uint32_t* tokens;
    std::size_t vocab_size;
    const float* logits; // [new_tokens][vocab_size]
};

} // namespace golden
