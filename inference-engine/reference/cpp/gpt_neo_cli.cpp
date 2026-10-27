// Greedy GPT-Neo generation from token IDs:
//   gpt_neo_cli --model DIR [--precision fp32|w8a8] [--new-tokens N] ID...
// DIR holds a Hugging Face config.json and model.safetensors. Prints the
// generated token IDs, space-separated, on one line.
#include "inference_engine/model.hpp"
#include "inference_engine/weights.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr const char* kUsage =
    "usage: gpt_neo_cli --model DIR [--precision fp32|w8a8] "
    "[--new-tokens N] ID...\n";

/// Parse a whole decimal argument, rejecting signs, junk and overflow.
std::optional<std::uint64_t> parse_count(const std::string& text) {
    if (text.empty() ||
        text.find_first_not_of("0123456789") != std::string::npos) {
        return std::nullopt;
    }
    try {
        return std::stoull(text);
    } catch (const std::out_of_range&) {
        return std::nullopt;
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    std::optional<std::string> model_dir;
    std::string precision = "fp32";
    std::uint64_t new_tokens = 20;
    inference_engine::TokenIds prompt;

    for (std::size_t index = 0; index < args.size(); ++index) {
        const std::string& arg = args[index];
        const bool has_value = index + 1 < args.size();
        if (arg == "--model" && has_value) {
            model_dir = args[++index];
        } else if (arg == "--precision" && has_value) {
            precision = args[++index];
        } else if (arg == "--new-tokens" && has_value) {
            const auto count = parse_count(args[++index]);
            if (!count) {
                std::cerr << "--new-tokens must be a non-negative integer\n";
                return 1;
            }
            new_tokens = *count;
        } else {
            const auto id = parse_count(arg);
            if (!id || *id > std::numeric_limits<std::uint32_t>::max()) {
                std::cerr << kUsage;
                return 1;
            }
            prompt.push_back(static_cast<std::uint32_t>(*id));
        }
    }
    if (!model_dir || prompt.empty() ||
        (precision != "fp32" && precision != "w8a8")) {
        std::cerr << kUsage;
        return 1;
    }

    try {
        inference_engine::ModelWeights<float> weights =
            inference_engine::load_gpt_neo(*model_dir);
        if (precision == "w8a8") {
            inference_engine::quantize_w8a8(weights);
        }
        inference_engine::GPTNeo<float> model(weights);
        const inference_engine::TokenIds generated =
            model.generate(prompt, new_tokens);
        for (std::size_t index = 0; index < generated.size(); ++index) {
            std::cout << (index == 0 ? "" : " ") << generated[index];
        }
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
