// Greedy GPT-Neo generation with the model split across chips:
//   tp_cli --model DIR [--precision fp32|w8a8] [--new-tokens N]
//          [--tp RANKS] [--scheme megatron|column] ID...
// Like gpt_neo_cli, but runs TpGptNeo: RANKS chips with tensor parallelism.
// Prints the generated token IDs, space-separated, on one line.
#include "dist_inference/tp_model.hpp"

#include "inference_engine/weights.hpp"

#include <algorithm>
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
    "usage: tp_cli --model DIR [--precision fp32|w8a8] [--new-tokens N] "
    "[--tp RANKS] [--scheme megatron|column] ID...\n";

/// The parsed command line.
struct Options {
    std::string model_dir;
    std::string precision = "fp32";
    std::uint64_t new_tokens = 20;
    std::uint64_t ranks = 1;
    dist_inference::TpScheme scheme = dist_inference::TpScheme::kMegatron;
    inference_engine::TokenIds prompt;
};

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

/// Parse a --scheme value.
std::optional<dist_inference::TpScheme> parse_scheme(const std::string& text) {
    if (text == "megatron") {
        return dist_inference::TpScheme::kMegatron;
    }
    if (text == "column") {
        return dist_inference::TpScheme::kColumnOnly;
    }
    return std::nullopt;
}

/// Parse a flag that takes a value, or print why its value is invalid.
/// @return false if the value is invalid.
bool parse_flag(const std::string& flag, const std::string& value,
                Options& options, std::optional<std::string>& model_dir) {
    if (flag == "--model") {
        model_dir = value;
    } else if (flag == "--precision") {
        options.precision = value;
    } else if (flag == "--new-tokens") {
        const auto count = parse_count(value);
        if (!count) {
            std::cerr << "--new-tokens must be a non-negative integer\n";
            return false;
        }
        options.new_tokens = *count;
    } else if (flag == "--tp") {
        const auto count = parse_count(value);
        if (!count || *count == 0) {
            std::cerr << "--tp must be a positive integer\n";
            return false;
        }
        options.ranks = *count;
    } else {
        const auto scheme = parse_scheme(value);
        if (!scheme) {
            std::cerr << "--scheme must be megatron or column\n";
            return false;
        }
        options.scheme = *scheme;
    }
    return true;
}

/// Parse the arguments after the program name, or print why they are invalid.
std::optional<Options> parse_args(const std::vector<std::string>& args) {
    const std::vector<std::string> flags{"--model", "--precision",
                                         "--new-tokens", "--tp", "--scheme"};
    Options options;
    std::optional<std::string> model_dir;
    for (std::size_t index = 0; index < args.size(); ++index) {
        const std::string& arg = args[index];
        const bool is_flag =
            std::find(flags.begin(), flags.end(), arg) != flags.end();
        if (is_flag && index + 1 < args.size()) {
            if (!parse_flag(arg, args[++index], options, model_dir)) {
                return std::nullopt;
            }
            continue;
        }
        const auto id = parse_count(arg);
        if (!id || *id > std::numeric_limits<std::uint32_t>::max()) {
            std::cerr << kUsage;
            return std::nullopt;
        }
        options.prompt.push_back(static_cast<std::uint32_t>(*id));
    }
    if (!model_dir || options.prompt.empty() ||
        (options.precision != "fp32" && options.precision != "w8a8")) {
        std::cerr << kUsage;
        return std::nullopt;
    }
    options.model_dir = *model_dir;
    return options;
}

} // namespace

int main(int argc, char** argv) {
    const std::optional<Options> options =
        parse_args(std::vector<std::string>(argv + 1, argv + argc));
    if (!options) {
        return 1;
    }

    try {
        inference_engine::ModelWeights<float> weights =
            inference_engine::load_gpt_neo(options->model_dir);
        if (options->precision == "w8a8") {
            inference_engine::quantize_w8a8(weights);
        }
        dist_inference::TpGptNeo<float> model(
            weights, static_cast<std::size_t>(options->ranks), options->scheme);
        const inference_engine::TokenIds generated =
            model.generate(options->prompt, options->new_tokens);
        for (std::size_t index = 0; index < generated.size(); ++index) {
            std::cout << (index == 0 ? "" : " ") << generated[index];
        }
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
