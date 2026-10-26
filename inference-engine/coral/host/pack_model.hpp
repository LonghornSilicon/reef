#pragma once

// Packs a host reference model into the byte layout the firmware reads
// (coral/include/reef_kernels/model_blob.hpp). For W8A8, the int8 weights
// and scales are the ones quantize_w8a8() stored in the model, so the
// firmware runs exactly the weights the reference ran.

#include "inference_engine/quant.hpp"
#include "inference_engine/tensor.hpp"
#include "inference_engine/weights.hpp"
#include "reef_kernels/model_blob.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace coral_host {

/// Appends aligned arrays to a growing blob and returns their offsets.
class BlobWriter {
  public:
    /// Reserves room for the header at offset 0.
    BlobWriter() : bytes_(sizeof(reef_kernels::BlobHeader)) { align(); }

    /// Appends raw values and returns where they start.
    template <typename T> std::uint32_t append(const std::vector<T>& values) {
        align();
        const auto offset = static_cast<std::uint32_t>(bytes_.size());
        bytes_.resize(bytes_.size() + (values.size() * sizeof(T)));
        std::memcpy(bytes_.data() + offset, values.data(),
                    values.size() * sizeof(T));
        return offset;
    }

    /// Appends a matrix transposed, [rows][columns] -> [columns][rows].
    template <typename T>
    std::uint32_t append_transposed(const inference_engine::Tensor<T>& matrix) {
        return append(inference_engine::transpose(matrix, 0, 1).values);
    }

    /// Overwrites bytes at an offset, for the header and layer table.
    template <typename T> void store(std::uint32_t offset, const T& value) {
        std::memcpy(bytes_.data() + offset, &value, sizeof(T));
    }

    /// Reserves zeroed room for count values of T and returns its offset.
    template <typename T> std::uint32_t reserve(std::size_t count) {
        return append(std::vector<T>(count));
    }

    std::vector<std::uint8_t> finish() {
        align();
        return std::move(bytes_);
    }

  private:
    void align() {
        constexpr std::size_t kAlign = 16;
        bytes_.resize((bytes_.size() + kAlign - 1) / kAlign * kAlign);
    }

    std::vector<std::uint8_t> bytes_;
};

/// The stored W8A8 copy of one projection, when packing W8A8.
using QuantizedRef = std::optional<
    std::reference_wrapper<const inference_engine::QuantizedWeight<float>>>;

/// Packs one projection: float weights, or the stored W8A8 copy.
inline reef_kernels::BlobLinear
pack_linear(BlobWriter& writer, const inference_engine::Tensor<float>& weight,
            QuantizedRef quantized) {
    if (!quantized.has_value()) {
        return {writer.append_transposed(weight), 0};
    }
    const inference_engine::QuantizedWeight<float>& stored = quantized->get();
    return {writer.append_transposed(stored.values),
            writer.append(stored.scales)};
}

/// Packs a whole model. W8A8 requires quantize_w8a8() to have run.
inline std::vector<std::uint8_t>
pack_gpt_neo(const inference_engine::ModelWeights<float>& weights,
             reef_kernels::Precision precision) {
    const inference_engine::GptNeoConfig& config = weights.config;
    const bool w8a8 = precision == reef_kernels::Precision::kW8A8;
    BlobWriter writer;

    reef_kernels::BlobHeader header{};
    header.magic = reef_kernels::kBlobMagic;
    header.version = reef_kernels::kBlobVersion;
    header.precision = precision;
    header.vocab_size = static_cast<std::uint32_t>(config.vocab_size);
    header.max_positions = static_cast<std::uint32_t>(config.max_positions);
    header.hidden_size = static_cast<std::uint32_t>(config.hidden_size);
    header.num_layers = static_cast<std::uint32_t>(config.num_layers);
    header.num_heads = static_cast<std::uint32_t>(config.num_heads);
    header.inner_size = static_cast<std::uint32_t>(config.inner_size);
    header.window_size = static_cast<std::uint32_t>(config.window_size);
    header.layer_norm_eps = config.layer_norm_eps;
    header.wte = writer.append(weights.wte.values);
    header.wpe = writer.append(weights.wpe.values);
    header.ln_f_weight = writer.append(weights.ln_f_weight.values);
    header.ln_f_bias = writer.append(weights.ln_f_bias.values);
    header.layers =
        writer.reserve<reef_kernels::BlobLayer>(weights.layers.size());

    for (std::size_t l = 0; l < weights.layers.size(); ++l) {
        const inference_engine::LayerWeights<float>& layer = weights.layers[l];
        const auto& attention = layer.attention;
        const auto& mlp = layer.mlp;
        if (w8a8 &&
            (!attention.quantized.has_value() || !mlp.quantized.has_value())) {
            throw std::invalid_argument(
                "pack_gpt_neo W8A8 needs quantize_w8a8() first");
        }
        const auto quantized = [&](const auto& stored,
                                   std::size_t index) -> QuantizedRef {
            if (!w8a8) {
                return std::nullopt;
            }
            return std::cref(stored->at(index));
        };

        reef_kernels::BlobLayer packed{};
        packed.ln_1_weight = writer.append(layer.ln_1_weight.values);
        packed.ln_1_bias = writer.append(layer.ln_1_bias.values);
        packed.query = pack_linear(writer, attention.query_weights,
                                   quantized(attention.quantized, 0));
        packed.key = pack_linear(writer, attention.key_weights,
                                 quantized(attention.quantized, 1));
        packed.value = pack_linear(writer, attention.value_weights,
                                   quantized(attention.quantized, 2));
        packed.output = pack_linear(writer, attention.output_weights,
                                    quantized(attention.quantized, 3));
        packed.output_bias = writer.append(attention.output_bias.values);
        packed.ln_2_weight = writer.append(layer.ln_2_weight.values);
        packed.ln_2_bias = writer.append(layer.ln_2_bias.values);
        packed.fc =
            pack_linear(writer, mlp.c_fc_weight, quantized(mlp.quantized, 0));
        packed.fc_bias = writer.append(mlp.c_fc_bias.values);
        packed.proj =
            pack_linear(writer, mlp.c_proj_weight, quantized(mlp.quantized, 1));
        packed.proj_bias = writer.append(mlp.c_proj_bias.values);
        packed.is_local = config.is_local(l) ? 1 : 0;
        writer.store(header.layers + static_cast<std::uint32_t>(
                                         l * sizeof(reef_kernels::BlobLayer)),
                     packed);
    }

    std::vector<std::uint8_t> blob = writer.finish();
    header.total_bytes = static_cast<std::uint32_t>(blob.size());
    std::memcpy(blob.data(), &header, sizeof header);
    return blob;
}

} // namespace coral_host
