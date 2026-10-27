#pragma once

/** @file
 *  @brief GPT-Neo configuration and weights: load, save, generate, quantize.
 *
 * Checkpoints use the Hugging Face layout: a config.json and a
 * model.safetensors whose tensor names start with "transformer." (as
 * GPTNeoForCausalLM saves them) or without that prefix (as GPTNeoModel does).
 * The language-model head is tied to the token embedding, so it is not
 * stored separately.
 */

#include "inference_engine/operator/attention.hpp"
#include "inference_engine/operator/mlp.hpp"
#include "inference_engine/quant.hpp"
#include "inference_engine/safetensors.hpp"
#include "inference_engine/tensor.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace inference_engine {

/** GPT-Neo hyperparameters. */
struct GptNeoConfig {
    /// Number of token IDs, the rows of the embedding table.
    std::size_t vocab_size{};
    /// Rows of the position table, the longest sequence the model accepts.
    std::size_t max_positions{};
    /// Model width.
    std::size_t hidden_size{};
    /// Transformer layers.
    std::size_t num_layers{};
    /// Attention heads per layer; must divide hidden_size.
    std::size_t num_heads{};
    /// MLP inner width, 4 * hidden_size in the published models.
    std::size_t inner_size{};
    /// Positions a local attention layer sees, including the query itself.
    std::size_t window_size{};
    /// Added to the variance in every LayerNorm.
    float layer_norm_eps{1e-5F};

    /** Whether a layer uses local (sliding-window) attention.
     *
     * GPT-Neo alternates global and local layers, starting with global.
     *
     * @param layer Zero-based layer index.
     * @return True for odd layers.
     */
    // A member, not static: callers ask the config, which owns the pattern.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    [[nodiscard]] bool is_local(std::size_t layer) const {
        return layer % 2 == 1;
    }
};

/** Parameters of one transformer layer.
 *
 * @tparam Scalar Tensor value type.
 */
template <typename Scalar> struct LayerWeights {
    /// LayerNorm gain before attention, [hidden].
    Tensor<Scalar> ln_1_weight;
    /// LayerNorm shift before attention, [hidden].
    Tensor<Scalar> ln_1_bias;
    /// Query, key and value projections (no bias) and the output projection.
    AttentionWeights<Scalar> attention{};
    /// LayerNorm gain before the MLP, [hidden].
    Tensor<Scalar> ln_2_weight;
    /// LayerNorm shift before the MLP, [hidden].
    Tensor<Scalar> ln_2_bias;
    /// Feed-forward block; both projections have biases.
    MlpWeights<Scalar> mlp{};
};

/** Every parameter of a GPT-Neo model.
 *
 * @tparam Scalar Tensor value type.
 */
template <typename Scalar> struct ModelWeights {
    /// Hyperparameters the tensors were shaped by.
    GptNeoConfig config;
    /// Token embedding [vocab, hidden], also the tied output projection.
    Tensor<Scalar> wte;
    /// Position embedding [max_positions, hidden].
    Tensor<Scalar> wpe;
    /// One entry per layer, in order.
    std::vector<LayerWeights<Scalar>> layers;
    /// Final LayerNorm gain, [hidden].
    Tensor<Scalar> ln_f_weight;
    /// Final LayerNorm shift, [hidden].
    Tensor<Scalar> ln_f_bias;
};

/** Throw unless a configuration describes a buildable model.
 *
 * @param config Hyperparameters to check.
 * @throws std::invalid_argument If a size is zero or the heads do not divide
 * the width.
 */
inline void validate_config(const GptNeoConfig& config) {
    if (config.vocab_size == 0 || config.max_positions == 0 ||
        config.hidden_size == 0 || config.num_layers == 0 ||
        config.num_heads == 0 || config.inner_size == 0 ||
        config.window_size == 0 || config.hidden_size % config.num_heads != 0) {
        throw std::invalid_argument(
            "GPT-Neo config sizes must be nonzero and the heads must divide "
            "the width");
    }
}

/** Parse a Hugging Face GPT-Neo config.json.
 *
 * @param text Contents of config.json.
 * @return The hyperparameters.
 * @throws std::invalid_argument If a field is missing or the model is not a
 * GPT-Neo this engine supports (gelu_new activation, layers alternating
 * global and local).
 */
inline GptNeoConfig parse_gpt_neo_config(std::string_view text) {
    const JsonValue json = parse_json(text);
    const auto size = [&json](const char* key) {
        return detail::json_size(json.at(key), key);
    };
    GptNeoConfig config;
    config.vocab_size = size("vocab_size");
    config.max_positions = size("max_position_embeddings");
    config.hidden_size = size("hidden_size");
    config.num_layers = size("num_layers");
    config.num_heads = size("num_heads");
    config.inner_size =
        json.contains("intermediate_size") &&
                json.at("intermediate_size").kind != JsonValue::Kind::kNull
            ? size("intermediate_size")
            : 4 * config.hidden_size;
    config.window_size = size("window_size");
    if (json.contains("layer_norm_epsilon")) {
        config.layer_norm_eps =
            static_cast<float>(json.at("layer_norm_epsilon").number);
    }
    if (json.contains("activation_function") &&
        json.at("activation_function").text != "gelu_new") {
        throw std::invalid_argument(
            "only the gelu_new activation is supported");
    }
    if (json.contains("attention_layers")) {
        const JsonValue& kinds = json.at("attention_layers");
        if (kinds.items.size() != config.num_layers) {
            throw std::invalid_argument(
                "attention_layers must name every layer");
        }
        for (std::size_t layer = 0; layer < config.num_layers; ++layer) {
            const char* expected = config.is_local(layer) ? "local" : "global";
            if (kinds.items[layer].text != expected) {
                throw std::invalid_argument(
                    "only alternating global and local layers are supported");
            }
        }
    }
    validate_config(config);
    return config;
}

namespace detail {

/** Remove one tensor from a loaded file, checking its shape.
 *
 * @param tensors Tensors read from the checkpoint.
 * @param name Full tensor name.
 * @param shape Shape the model requires.
 * @return The tensor.
 * @throws std::invalid_argument If the tensor is missing or misshapen.
 */
inline Tensor<float> take_tensor(std::map<std::string, Tensor<float>>& tensors,
                                 const std::string& name,
                                 const std::vector<std::size_t>& shape) {
    auto found = tensors.find(name);
    if (found == tensors.end()) {
        throw std::invalid_argument("checkpoint has no F32 tensor " + name);
    }
    if (found->second.shape != shape) {
        throw std::invalid_argument("checkpoint tensor " + name +
                                    " has the wrong shape");
    }
    Tensor<float> tensor = std::move(found->second);
    tensors.erase(found);
    return tensor;
}

/** Hugging Face name of every tensor in one layer, in LayerWeights order.
 *
 * @param prefix "transformer." or empty.
 * @param layer Zero-based layer index.
 * @return Names of ln_1 weight and bias; q, k, v, out weights; out bias;
 * ln_2 weight and bias; c_fc weight and bias; c_proj weight and bias.
 */
inline std::array<std::string, 13> layer_tensor_names(const std::string& prefix,
                                                      std::size_t layer) {
    const std::string base = prefix + "h." + std::to_string(layer) + ".";
    const std::string attention = base + "attn.attention.";
    return {base + "ln_1.weight",        base + "ln_1.bias",
            attention + "q_proj.weight", attention + "k_proj.weight",
            attention + "v_proj.weight", attention + "out_proj.weight",
            attention + "out_proj.bias", base + "ln_2.weight",
            base + "ln_2.bias",          base + "mlp.c_fc.weight",
            base + "mlp.c_fc.bias",      base + "mlp.c_proj.weight",
            base + "mlp.c_proj.bias"};
}

/** Uniform values from a 32-bit linear congruential generator.
 *
 * The same seed gives the same values on every platform, unlike the
 * standard library's distributions.
 */
class Lcg {
  public:
    /** Start the sequence.
     *
     * @param seed Initial state.
     */
    explicit Lcg(std::uint32_t seed) : state_(seed) {}

    /** Draw the next value.
     *
     * @return A value in [-1, 1).
     */
    float next() {
        constexpr std::uint32_t kMultiplier = 1664525U;
        constexpr std::uint32_t kIncrement = 1013904223U;
        constexpr std::uint32_t kDroppedBits = 8;
        constexpr float kInverse2To24 = 1.0F / 16777216.0F;
        state_ = (state_ * kMultiplier) + kIncrement;
        // The top 24 bits fill a float mantissa exactly.
        const float unit =
            static_cast<float>(state_ >> kDroppedBits) * kInverse2To24;
        return (2.0F * unit) - 1.0F;
    }

    /** Fill a tensor with values in [center - spread, center + spread).
     *
     * @param shape Tensor dimensions.
     * @param center Middle of the range.
     * @param spread Half-width of the range.
     * @return The filled tensor.
     */
    Tensor<float> tensor(const std::vector<std::size_t>& shape, float center,
                         float spread) {
        Tensor<float> result{shape, std::vector<float>(element_count(shape))};
        for (float& value : result.values) {
            value = center + (spread * next());
        }
        return result;
    }

  private:
    std::uint32_t state_;
};

} // namespace detail

/** Load a Hugging Face GPT-Neo checkpoint.
 *
 * Float only: the loader reads F32 safetensors.
 *
 * @param model_dir Directory holding config.json and model.safetensors.
 * @return Float weights; nothing is quantized.
 * @throws std::runtime_error If a file cannot be read.
 * @throws std::invalid_argument If the config is unsupported or a tensor is
 * missing or misshapen.
 */
inline ModelWeights<float>
load_gpt_neo(const std::filesystem::path& model_dir) {
    ModelWeights<float> weights;
    weights.config =
        parse_gpt_neo_config(detail::read_file(model_dir / "config.json"));
    const GptNeoConfig& config = weights.config;
    std::map<std::string, Tensor<float>> tensors =
        read_safetensors(model_dir / "model.safetensors");
    const std::string prefix =
        tensors.count("transformer.wte.weight") != 0 ? "transformer." : "";

    const std::size_t hidden = config.hidden_size;
    const std::size_t inner = config.inner_size;
    const std::vector<std::size_t> row{hidden};
    const std::vector<std::size_t> square{hidden, hidden};
    const auto take = [&](const std::string& name,
                          const std::vector<std::size_t>& shape) {
        return detail::take_tensor(tensors, name, shape);
    };

    weights.wte = take(prefix + "wte.weight", {config.vocab_size, hidden});
    weights.wpe = take(prefix + "wpe.weight", {config.max_positions, hidden});
    for (std::size_t layer = 0; layer < config.num_layers; ++layer) {
        const auto names = detail::layer_tensor_names(prefix, layer);
        LayerWeights<float> loaded{};
        loaded.ln_1_weight = take(names[0], row);
        loaded.ln_1_bias = take(names[1], row);
        loaded.attention.num_heads = config.num_heads;
        loaded.attention.query_weights = take(names[2], square);
        loaded.attention.key_weights = take(names[3], square);
        loaded.attention.value_weights = take(names[4], square);
        loaded.attention.output_weights = take(names[5], square);
        loaded.attention.output_bias = take(names[6], row);
        loaded.ln_2_weight = take(names[7], row);
        loaded.ln_2_bias = take(names[8], row);
        loaded.mlp.input_size = hidden;
        loaded.mlp.inner_size = inner;
        loaded.mlp.c_fc_weight = take(names[9], {inner, hidden});
        loaded.mlp.c_fc_bias = take(names[10], {inner});
        loaded.mlp.c_proj_weight = take(names[11], {hidden, inner});
        loaded.mlp.c_proj_bias = take(names[12], row);
        weights.layers.push_back(std::move(loaded));
    }
    weights.ln_f_weight = take(prefix + "ln_f.weight", row);
    weights.ln_f_bias = take(prefix + "ln_f.bias", row);
    assert(weights.layers.size() == config.num_layers);
    return weights;
}

/** Save float weights as a Hugging Face GPT-Neo checkpoint.
 *
 * Writes config.json and model.safetensors (names prefixed "transformer.")
 * that load_gpt_neo() reads back exactly. Quantized copies are not saved;
 * call quantize_w8a8() again after loading.
 *
 * @param weights Model to save.
 * @param model_dir Directory to create or overwrite files in.
 * @throws std::runtime_error If a file cannot be written.
 */
inline void save_gpt_neo(const ModelWeights<float>& weights,
                         const std::filesystem::path& model_dir) {
    const GptNeoConfig& config = weights.config;
    std::filesystem::create_directories(model_dir);

    std::ofstream json(model_dir / "config.json");
    json.precision(std::numeric_limits<float>::max_digits10);
    json << "{\n  \"model_type\": \"gpt_neo\",\n"
         << "  \"activation_function\": \"gelu_new\",\n"
         << "  \"vocab_size\": " << config.vocab_size << ",\n"
         << "  \"max_position_embeddings\": " << config.max_positions << ",\n"
         << "  \"hidden_size\": " << config.hidden_size << ",\n"
         << "  \"num_layers\": " << config.num_layers << ",\n"
         << "  \"num_heads\": " << config.num_heads << ",\n"
         << "  \"intermediate_size\": " << config.inner_size << ",\n"
         << "  \"window_size\": " << config.window_size << ",\n"
         << "  \"layer_norm_epsilon\": " << config.layer_norm_eps << ",\n"
         << "  \"attention_layers\": [";
    for (std::size_t layer = 0; layer < config.num_layers; ++layer) {
        json << (layer == 0 ? "" : ", ")
             << (config.is_local(layer) ? "\"local\"" : "\"global\"");
    }
    json << "]\n}\n";
    if (!json) {
        throw std::runtime_error("cannot write config.json in " +
                                 model_dir.string());
    }

    const std::string prefix = "transformer.";
    std::map<std::string, Tensor<float>> tensors;
    tensors[prefix + "wte.weight"] = weights.wte;
    tensors[prefix + "wpe.weight"] = weights.wpe;
    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        const LayerWeights<float>& source = weights.layers[layer];
        const auto names = detail::layer_tensor_names(prefix, layer);
        tensors[names[0]] = source.ln_1_weight;
        tensors[names[1]] = source.ln_1_bias;
        tensors[names[2]] = source.attention.query_weights;
        tensors[names[3]] = source.attention.key_weights;
        tensors[names[4]] = source.attention.value_weights;
        tensors[names[5]] = source.attention.output_weights;
        tensors[names[6]] = source.attention.output_bias;
        tensors[names[7]] = source.ln_2_weight;
        tensors[names[8]] = source.ln_2_bias;
        tensors[names[9]] = source.mlp.c_fc_weight;
        tensors[names[10]] = source.mlp.c_fc_bias;
        tensors[names[11]] = source.mlp.c_proj_weight;
        tensors[names[12]] = source.mlp.c_proj_bias;
    }
    tensors[prefix + "ln_f.weight"] = weights.ln_f_weight;
    tensors[prefix + "ln_f.bias"] = weights.ln_f_bias;
    write_safetensors(model_dir / "model.safetensors", tensors);
}

/** Build a model with small pseudo-random weights, for tests and kernels.
 *
 * Every tensor is drawn in a fixed order (wte, wpe, then each layer in
 * LayerWeights field order, then ln_f) from one portable generator, so a
 * seed gives the same model on every platform. Projections, biases and
 * embeddings are uniform in [-0.1, 0.1); LayerNorm gains are uniform in
 * [0.9, 1.1).
 *
 * @param config Hyperparameters to shape the tensors by.
 * @param seed Generator seed.
 * @return Float weights; nothing is quantized.
 * @throws std::invalid_argument If @p config is invalid.
 */
inline ModelWeights<float> random_gpt_neo(const GptNeoConfig& config,
                                          std::uint32_t seed) {
    validate_config(config);
    constexpr float kSpread = 0.1F;
    detail::Lcg lcg(seed);
    const std::size_t hidden = config.hidden_size;
    const std::size_t inner = config.inner_size;
    const auto values = [&lcg](const std::vector<std::size_t>& shape) {
        return lcg.tensor(shape, 0.0F, kSpread);
    };
    const auto gains = [&lcg, hidden]() {
        return lcg.tensor({hidden}, 1.0F, kSpread);
    };

    ModelWeights<float> weights;
    weights.config = config;
    weights.wte = values({config.vocab_size, hidden});
    weights.wpe = values({config.max_positions, hidden});
    for (std::size_t layer = 0; layer < config.num_layers; ++layer) {
        LayerWeights<float> drawn{};
        drawn.ln_1_weight = gains();
        drawn.ln_1_bias = values({hidden});
        drawn.attention.num_heads = config.num_heads;
        drawn.attention.query_weights = values({hidden, hidden});
        drawn.attention.key_weights = values({hidden, hidden});
        drawn.attention.value_weights = values({hidden, hidden});
        drawn.attention.output_weights = values({hidden, hidden});
        drawn.attention.output_bias = values({hidden});
        drawn.ln_2_weight = gains();
        drawn.ln_2_bias = values({hidden});
        drawn.mlp.input_size = hidden;
        drawn.mlp.inner_size = inner;
        drawn.mlp.c_fc_weight = values({inner, hidden});
        drawn.mlp.c_fc_bias = values({inner});
        drawn.mlp.c_proj_weight = values({hidden, inner});
        drawn.mlp.c_proj_bias = values({hidden});
        weights.layers.push_back(std::move(drawn));
    }
    weights.ln_f_weight = gains();
    weights.ln_f_bias = values({hidden});
    return weights;
}

/** Add W8A8 copies of every attention and MLP projection.
 *
 * Each weight is quantized row by row with quantize_rows(). The token
 * embedding, and so the tied output projection, stays in Scalar.
 *
 * @tparam Scalar Floating-point tensor value type.
 * @param weights Model to quantize in place.
 * @throws std::invalid_argument If a projection weight is not a valid matrix
 * or holds a non-finite value.
 */
template <typename Scalar> void quantize_w8a8(ModelWeights<Scalar>& weights) {
    for (LayerWeights<Scalar>& layer : weights.layers) {
        AttentionWeights<Scalar>& attention = layer.attention;
        attention.quantized = std::array<QuantizedWeight<Scalar>, 4>{
            quantize_rows(attention.query_weights),
            quantize_rows(attention.key_weights),
            quantize_rows(attention.value_weights),
            quantize_rows(attention.output_weights)};
        layer.mlp.quantized = std::array<QuantizedWeight<Scalar>, 2>{
            quantize_rows(layer.mlp.c_fc_weight),
            quantize_rows(layer.mlp.c_proj_weight)};
    }
}

} // namespace inference_engine
