#pragma once

// Byte layout of a packed GPT-Neo model: what the host loads into memory for
// the firmware. coral/host/pack_model.hpp writes it from the host reference's
// ModelWeights (after quantize_w8a8() for W8A8), and coral/firmware/ reads
// it. Both sides include this header, so it is plain fixed-width data:
// little-endian, every offset in bytes from the start of the blob and
// 16-byte aligned.
//
// Matrices that a linear kernel reads are stored transposed, [in][out] (see
// kernels.hpp). The token embedding stays [vocab][hidden]: rows serve the
// embedding lookup and, through matvec_rows_f32, the tied lm_head.

#include <cstdint>

namespace reef_kernels {

inline constexpr std::uint32_t kBlobMagic = 0x46454552; // "REEF"
inline constexpr std::uint32_t kBlobVersion = 1;

// How the attention and MLP projections are stored and run. Embeddings,
// LayerNorm, attention scores, GELU and lm_head are float in both.
enum class Precision : std::uint32_t {
    kFloat32 = 0, // float weights, linear_f32
    kW8A8 = 1,    // int8 weights with row scales, linear_w8a8
};

// One projection. kFloat32: weight is float [in][out] and scales is unused.
// kW8A8: weight is int8 [in][out] and scales is float [out].
struct BlobLinear {
    std::uint32_t weight;
    std::uint32_t scales;
};

struct BlobLayer {
    std::uint32_t ln_1_weight, ln_1_bias; // float [hidden]
    BlobLinear query, key, value;         // [hidden][hidden], no bias
    BlobLinear output;                    // [hidden][hidden]
    std::uint32_t output_bias;            // float [hidden]
    std::uint32_t ln_2_weight, ln_2_bias; // float [hidden]
    BlobLinear fc;                        // [hidden][inner]
    std::uint32_t fc_bias;                // float [inner]
    BlobLinear proj;                      // [inner][hidden]
    std::uint32_t proj_bias;              // float [hidden]
    std::uint32_t is_local;               // 1: sliding-window attention
};

struct BlobHeader {
    std::uint32_t magic;
    std::uint32_t version;
    Precision precision;
    std::uint32_t vocab_size;
    std::uint32_t max_positions;
    std::uint32_t hidden_size;
    std::uint32_t num_layers;
    std::uint32_t num_heads;
    std::uint32_t inner_size;
    std::uint32_t window_size;
    float layer_norm_eps;
    std::uint32_t wte;         // float [vocab][hidden]
    std::uint32_t wpe;         // float [max_positions][hidden]
    std::uint32_t ln_f_weight; // float [hidden]
    std::uint32_t ln_f_bias;   // float [hidden]
    std::uint32_t layers;      // BlobLayer [num_layers]
    std::uint32_t total_bytes;
};

} // namespace reef_kernels
