#pragma once

// Packs one rank's share of a GPT-Neo model (dist_coral/tp_blob.hpp). The
// shares are cut with dist_inference's shard_range and slicing functions,
// the same calls TpGptNeo makes, so each chip holds exactly the weights its
// rank in the reference runs. For W8A8, the int8 weights and scales are the
// ones quantize_w8a8() stored; a slice is cut from them, never requantized.

#include "dist_coral/tp_blob.hpp"
#include "dist_inference/shard.hpp"

#include "inference_engine/weights.hpp"
#include "pack_model.hpp"
#include "reef_kernels/model_blob.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace dist_coral_host {

using dist_inference::Range;
using dist_inference::shard_range;
using inference_engine::QuantizedWeight;
using inference_engine::Tensor;

/// Which part of a weight [out, in] a rank holds.
enum class Cut {
    kOutputs, ///< Output rows: a column-parallel share.
    kInputs,  ///< Input columns: a row-parallel share.
};

/// Packs a rank's share of one projection: float, or the stored W8A8 copy.
inline reef_kernels::BlobLinear
pack_share(coral_host::BlobWriter& writer, const Tensor<float>& weight,
           const QuantizedWeight<float>* quantized, Cut cut, Range range) {
    if (quantized == nullptr) {
        return coral_host::pack_linear(
            writer,
            cut == Cut::kOutputs ? dist_inference::slice_rows(weight, range)
                                 : dist_inference::slice_columns(weight, range),
            std::nullopt);
    }
    const QuantizedWeight<float> share =
        cut == Cut::kOutputs ? dist_inference::slice_rows(*quantized, range)
                             : dist_inference::slice_columns(*quantized, range);
    return coral_host::pack_linear(writer, weight, std::cref(share));
}

/// Packs rank @p rank of @p ranks. W8A8 requires quantize_w8a8() to have run.
inline std::vector<std::uint8_t>
pack_rank(const inference_engine::ModelWeights<float>& weights,
          reef_kernels::Precision precision, std::size_t ranks,
          std::size_t rank) {
    const inference_engine::GptNeoConfig& config = weights.config;
    if (ranks == 0 || config.num_heads % ranks != 0 ||
        config.inner_size % ranks != 0) {
        throw std::invalid_argument(
            "tensor-parallel ranks must divide the heads and the MLP width");
    }
    const bool w8a8 = precision == reef_kernels::Precision::kW8A8;
    const std::size_t head_dim = config.hidden_size / config.num_heads;
    const Range heads = shard_range(config.num_heads, ranks, rank);
    const Range columns{heads.start * head_dim, heads.count * head_dim};
    const Range inner = shard_range(config.inner_size, ranks, rank);
    const Range vocab = shard_range(config.vocab_size, ranks, rank);
    coral_host::BlobWriter writer(sizeof(dist_coral::RankHeader));

    dist_coral::RankHeader header{};
    header.magic = dist_coral::kRankBlobMagic;
    header.version = dist_coral::kRankBlobVersion;
    header.precision = precision;
    header.rank = static_cast<std::uint32_t>(rank);
    header.ranks = static_cast<std::uint32_t>(ranks);
    header.max_positions = static_cast<std::uint32_t>(config.max_positions);
    header.hidden_size = static_cast<std::uint32_t>(config.hidden_size);
    header.num_layers = static_cast<std::uint32_t>(config.num_layers);
    header.head_dim = static_cast<std::uint32_t>(head_dim);
    header.local_heads = static_cast<std::uint32_t>(heads.count);
    header.local_inner = static_cast<std::uint32_t>(inner.count);
    header.vocab_start = static_cast<std::uint32_t>(vocab.start);
    header.local_vocab = static_cast<std::uint32_t>(vocab.count);
    header.window_size = static_cast<std::uint32_t>(config.window_size);
    header.layer_norm_eps = config.layer_norm_eps;
    header.ln_f_weight = writer.append(weights.ln_f_weight.values);
    header.ln_f_bias = writer.append(weights.ln_f_bias.values);
    header.lm_head =
        writer.append(dist_inference::slice_rows(weights.wte, vocab).values);
    header.layers =
        writer.reserve<dist_coral::RankLayer>(weights.layers.size());

    for (std::size_t l = 0; l < weights.layers.size(); ++l) {
        const inference_engine::LayerWeights<float>& layer = weights.layers[l];
        const auto& attention = layer.attention;
        const auto& mlp = layer.mlp;
        if (w8a8 &&
            (!attention.quantized.has_value() || !mlp.quantized.has_value())) {
            throw std::invalid_argument(
                "pack_rank W8A8 needs quantize_w8a8() first");
        }
        const auto quantized =
            [w8a8](const auto& stored,
                   std::size_t index) -> const QuantizedWeight<float>* {
            return w8a8 ? &stored->at(index) : nullptr;
        };

        dist_coral::RankLayer packed{};
        packed.ln_1_weight = writer.append(layer.ln_1_weight.values);
        packed.ln_1_bias = writer.append(layer.ln_1_bias.values);
        packed.query = pack_share(writer, attention.query_weights,
                                  quantized(attention.quantized, 0),
                                  Cut::kOutputs, columns);
        packed.key = pack_share(writer, attention.key_weights,
                                quantized(attention.quantized, 1),
                                Cut::kOutputs, columns);
        packed.value = pack_share(writer, attention.value_weights,
                                  quantized(attention.quantized, 2),
                                  Cut::kOutputs, columns);
        packed.output = pack_share(writer, attention.output_weights,
                                   quantized(attention.quantized, 3),
                                   Cut::kInputs, columns);
        packed.ln_2_weight = writer.append(layer.ln_2_weight.values);
        packed.ln_2_bias = writer.append(layer.ln_2_bias.values);
        packed.fc =
            pack_share(writer, mlp.c_fc_weight, quantized(mlp.quantized, 0),
                       Cut::kOutputs, inner);
        packed.fc_bias = writer.append(
            dist_inference::slice_vector(mlp.c_fc_bias, inner).values);
        packed.proj =
            pack_share(writer, mlp.c_proj_weight, quantized(mlp.quantized, 1),
                       Cut::kInputs, inner);
        packed.is_local = config.is_local(l) ? 1 : 0;
        writer.store(header.layers + static_cast<std::uint32_t>(
                                         l * sizeof(dist_coral::RankLayer)),
                     packed);
    }

    std::vector<std::uint8_t> blob = writer.finish();
    header.total_bytes = static_cast<std::uint32_t>(blob.size());
    std::memcpy(blob.data(), &header, sizeof header);
    return blob;
}

} // namespace dist_coral_host
