// GPT-Neo model tests on a tiny random configuration: output shapes,
// bit-identical prefill and decode in both precisions, greedy generation,
// the local attention window, W8A8 accuracy, and the checkpoint loader.
#include "inference_engine/model.hpp"
#include "inference_engine/operator/block.hpp"
#include "inference_engine/safetensors.hpp"
#include "inference_engine/weights.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace inference_engine {

namespace {

// vocab 64, 32 positions, width 32, 2 layers (global then local), 4 heads,
// MLP width 128, window 4.
constexpr GptNeoConfig kTiny{64, 32, 32, 2, 4, 128, 4, 1e-5F};
constexpr std::uint32_t kSeed = 7;

const TokenIds prompt_tokens{3, 17, 42, 5, 63, 0, 9, 9, 28, 51};

ModelWeights<float> tiny_weights(bool w8a8) {
    ModelWeights<float> weights = random_gpt_neo(kTiny, kSeed);
    if (w8a8) {
        quantize_w8a8(weights);
    }
    return weights;
}

/// Logits after each prefix of the prompt, fed one token at a time.
std::vector<Tensor<float>> decode_logits(const ModelWeights<float>& weights) {
    GPTNeo<float> model(weights);
    std::vector<Tensor<float>> logits;
    for (const std::uint32_t id : prompt_tokens) {
        logits.push_back(model.forward({id}));
    }
    return logits;
}

void expect_prefill_matches_decode(bool w8a8) {
    const ModelWeights<float> weights = tiny_weights(w8a8);
    const std::vector<Tensor<float>> decoded = decode_logits(weights);
    for (std::size_t length = 1; length <= prompt_tokens.size(); ++length) {
        GPTNeo<float> model(weights);
        const Tensor<float> prefilled = model.forward(TokenIds(
            prompt_tokens.begin(),
            prompt_tokens.begin() + static_cast<std::ptrdiff_t>(length)));
        // Bitwise, not approximately: every operator is row-independent.
        EXPECT_EQ(prefilled.values, decoded[length - 1].values) << length;
    }
}

/// Deterministic rows [rows, hidden] in [-1, 1).
Tensor<float> pattern_rows(std::size_t rows, std::size_t hidden) {
    std::vector<float> values(rows * hidden);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = (static_cast<float>((i * 29) % 97) / 48.5F) - 1.0F;
    }
    return Tensor<float>{{rows, hidden}, std::move(values)};
}

std::filesystem::path scratch_dir(const std::string& name) {
    const std::filesystem::path dir =
        std::filesystem::path(::testing::TempDir()) / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

} // namespace

// A prompt yields one finite logit per vocabulary entry for the next token.
TEST(GptNeoTest, ForwardReturnsOneRowOfLogits) {
    const ModelWeights<float> weights = tiny_weights(false);
    GPTNeo<float> model(weights);

    const Tensor<float> logits = model.forward(prompt_tokens);

    ASSERT_EQ(logits.shape, std::vector<std::size_t>({1, kTiny.vocab_size}));
    for (const float value : logits.values) {
        EXPECT_TRUE(std::isfinite(value));
    }
}

// Feeding the prompt at once or one token at a time gives the same float
// logits, bit for bit.
TEST(GptNeoTest, PrefillMatchesDecodeBitwiseInFloat) {
    expect_prefill_matches_decode(false);
}

// The same prefill and decode equivalence holds when the projections run in
// W8A8.
TEST(GptNeoTest, PrefillMatchesDecodeBitwiseInW8A8) {
    expect_prefill_matches_decode(true);
}

// generate() picks the argmax at every step, repeats across calls, and returns
// nothing for zero tokens.
TEST(GptNeoTest, GenerateIsGreedyAndRepeatable) {
    const ModelWeights<float> weights = tiny_weights(false);
    GPTNeo<float> model(weights);

    const TokenIds first = model.generate(prompt_tokens, 8);
    const TokenIds second = model.generate(prompt_tokens, 8);

    ASSERT_EQ(first.size(), 8U);
    EXPECT_EQ(first, second);
    // The same tokens, chosen by hand from the logits.
    model.reset();
    Tensor<float> logits = model.forward(prompt_tokens);
    for (const std::uint32_t expected : first) {
        const auto chosen = static_cast<std::uint32_t>(argmax(logits));
        EXPECT_EQ(chosen, expected);
        logits = model.forward({chosen});
    }
    EXPECT_TRUE(model.generate(prompt_tokens, 0).empty());
}

// argmax() breaks ties toward the lower index and rejects an empty tensor.
TEST(GptNeoTest, ArgmaxPicksTheFirstOfEqualMaxima) {
    const Tensor<float> row{{1, 5}, {0.5F, 2.0F, -1.0F, 2.0F, 1.0F}};
    EXPECT_EQ(argmax(row), 1U);
    EXPECT_THROW(argmax(Tensor<float>{}), std::invalid_argument);
}

// A local layer's last row ignores a change outside its window; a global
// layer's does not.
TEST(GptNeoTest, LocalLayerIgnoresPositionsOutsideItsWindow) {
    const ModelWeights<float> weights = tiny_weights(false);
    const std::size_t rows = 8;
    const Tensor<float> input = pattern_rows(rows, kTiny.hidden_size);
    Tensor<float> changed = input;
    for (std::size_t column = 0; column < kTiny.hidden_size; ++column) {
        changed.values[column] += 0.5F; // row 0 only
    }
    const std::size_t last = (rows - 1) * kTiny.hidden_size;
    const auto last_row = [&](const Tensor<float>& output) {
        return std::vector<float>(output.values.begin() +
                                      static_cast<std::ptrdiff_t>(last),
                                  output.values.end());
    };

    // Layer 1 is local with window 4: row 7 sees rows 4 to 7 only.
    ASSERT_TRUE(kTiny.is_local(1));
    Block<float> local(weights.layers[1], kTiny, 1);
    const auto local_before = last_row(local.forward(input));
    local.clear_cache();
    EXPECT_EQ(last_row(local.forward(changed)), local_before);

    // Layer 0 is global, so the same change reaches row 7.
    Block<float> global(weights.layers[0], kTiny, 0);
    const auto global_before = last_row(global.forward(input));
    global.clear_cache();
    EXPECT_NE(last_row(global.forward(changed)), global_before);
}

// W8A8 logits differ from float logits, but by under 5% of the largest logit.
TEST(GptNeoTest, W8A8LogitsStayCloseToFloat) {
    const ModelWeights<float> float_weights = tiny_weights(false);
    const ModelWeights<float> int8_weights = tiny_weights(true);
    GPTNeo<float> float_model(float_weights);
    GPTNeo<float> int8_model(int8_weights);

    const Tensor<float> exact = float_model.forward(prompt_tokens);
    const Tensor<float> quantized = int8_model.forward(prompt_tokens);

    float largest = 0.0F;
    float worst = 0.0F;
    for (std::size_t i = 0; i < exact.values.size(); ++i) {
        largest = std::max(largest, std::fabs(exact.values[i]));
        worst =
            std::max(worst, std::fabs(exact.values[i] - quantized.values[i]));
    }
    EXPECT_GT(worst, 0.0F); // the int8 path really ran
    EXPECT_LT(worst, 0.05F * largest);
}

// forward() rejects an empty input, a sequence past the position table, and an
// unknown token ID.
TEST(GptNeoTest, ForwardRejectsPositionsPastTheTable) {
    const ModelWeights<float> weights = tiny_weights(false);
    GPTNeo<float> model(weights);

    EXPECT_THROW(model.forward({}), std::invalid_argument);
    EXPECT_THROW(model.forward(TokenIds(kTiny.max_positions + 1, 1)),
                 std::invalid_argument);
    EXPECT_THROW(model.forward({static_cast<std::uint32_t>(kTiny.vocab_size)}),
                 std::invalid_argument);
}

// Saving then loading a checkpoint keeps every value and the generated tokens.
TEST(GptNeoTest, CheckpointRoundTripsExactly) {
    const ModelWeights<float> weights = tiny_weights(false);
    const std::filesystem::path dir = scratch_dir("gpt_neo_round_trip");

    save_gpt_neo(weights, dir);
    const ModelWeights<float> loaded = load_gpt_neo(dir);

    EXPECT_EQ(loaded.config.hidden_size, kTiny.hidden_size);
    EXPECT_EQ(loaded.config.inner_size, kTiny.inner_size);
    EXPECT_EQ(loaded.config.window_size, kTiny.window_size);
    EXPECT_EQ(loaded.config.layer_norm_eps, kTiny.layer_norm_eps);
    EXPECT_EQ(loaded.wte.values, weights.wte.values);
    EXPECT_EQ(loaded.layers[1].mlp.c_proj_weight.values,
              weights.layers[1].mlp.c_proj_weight.values);
    GPTNeo<float> original(weights);
    GPTNeo<float> reloaded(loaded);
    EXPECT_EQ(reloaded.generate(prompt_tokens, 6),
              original.generate(prompt_tokens, 6));
}

// Loading a checkpoint with a tensor removed throws instead of running with
// missing weights.
TEST(GptNeoTest, LoaderReportsAMissingTensor) {
    const std::filesystem::path dir = scratch_dir("gpt_neo_missing");
    save_gpt_neo(tiny_weights(false), dir);
    std::map<std::string, Tensor<float>> tensors =
        read_safetensors(dir / "model.safetensors");
    tensors.erase("transformer.h.1.mlp.c_fc.bias");
    write_safetensors(dir / "model.safetensors", tensors);

    EXPECT_THROW(load_gpt_neo(dir), std::invalid_argument);
}

// config.json fields map onto GptNeoConfig, a null intermediate size means 4 *
// hidden, and a local-first layer order is rejected.
TEST(GptNeoTest, ConfigFollowsTheHuggingFaceFields) {
    const GptNeoConfig config = parse_gpt_neo_config(R"({
        "activation_function": "gelu_new",
        "attention_layers": ["global", "local", "global", "local"],
        "hidden_size": 64, "intermediate_size": null,
        "layer_norm_epsilon": 1e-05, "max_position_embeddings": 2048,
        "num_heads": 16, "num_layers": 4, "vocab_size": 50257,
        "window_size": 256, "summary_activation": null,
        "_name_or_path": "a \"quoted\" é path"})");

    EXPECT_EQ(config.inner_size, 256U); // null means 4 * hidden
    EXPECT_EQ(config.max_positions, 2048U);
    EXPECT_EQ(config.layer_norm_eps, 1e-5F);
    EXPECT_THROW(parse_gpt_neo_config(R"({
        "attention_layers": ["local", "global"], "hidden_size": 64,
        "max_position_embeddings": 8, "num_heads": 4, "num_layers": 2,
        "vocab_size": 8, "window_size": 4})"),
                 std::invalid_argument);
}

// The JSON parser reads arrays, numbers, literals and escapes, and rejects
// trailing text and missing values.
TEST(GptNeoTest, JsonParserHandlesEscapesAndRejectsJunk) {
    const JsonValue value =
        parse_json(R"({"a": [1, -2.5e1, true, null], "b": "x\né"})");

    EXPECT_EQ(value.at("a").items[1].number, -25.0);
    EXPECT_TRUE(value.at("a").items[2].boolean);
    EXPECT_EQ(value.at("b").text, "x\n\xc3\xa9");
    EXPECT_THROW(parse_json(R"({"a": 1} extra)"), std::invalid_argument);
    EXPECT_THROW(parse_json(R"({"a": })"), std::invalid_argument);
}

} // namespace inference_engine
