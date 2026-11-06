// Teacher-forced comparison of TpGptNeo with one chip (Test B):
//   tp_compare --model DIR --tokens FILE [--precision fp32|w8a8]
//              [--scheme megatron|column] [--tp N,N,...]
//              [--reference same|fp32]
// FILE holds one token sequence per line. Each sequence is fed one token at
// a time, the same tokens to every model whatever they predict, and the
// logits are compared at every position. The reference is one chip at the
// same precision, or with --reference fp32 one chip in fp32: then the rows
// show how far each split W8A8 model is from full precision, and rank count
// 1 (one chip in W8A8) is the error W8A8 already adds. For each rank count
// it prints one tab-separated row:
//   precision scheme ranks reference positions max_rel mean_rel
//   argmax_agree ppl_reference ppl_split
// max_rel and mean_rel are the largest logit difference over the largest
// reference logit, worst and mean over positions; argmax_agree is the
// fraction of positions with the same top token; the perplexities score
// each sequence's real next token.
#include "dist_inference/tp_model.hpp"

#include "inference_engine/model.hpp"
#include "inference_engine/weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using inference_engine::GPTNeo;
using inference_engine::ModelWeights;
using inference_engine::Tensor;
using inference_engine::TokenIds;

constexpr const char* kUsage =
    "usage: tp_compare --model DIR --tokens FILE [--precision fp32|w8a8] "
    "[--scheme megatron|column] [--tp N,N,...] [--reference same|fp32]\n";

/// The parsed command line.
struct Options {
    std::string model_dir;
    std::string tokens_file;
    std::string precision = "fp32";
    std::string scheme = "megatron";
    std::vector<std::size_t> ranks{2, 4};
    std::string reference = "same";
};

/// Running totals for one compared model.
struct Totals {
    std::size_t positions = 0;
    double max_rel = 0.0;
    double sum_rel = 0.0;
    std::size_t agree = 0;
    double nll_reference = 0.0;
    double nll_split = 0.0;
    std::size_t scored = 0;
};

/// Parse "2,4,8" into rank counts, or nothing if any entry is invalid.
std::optional<std::vector<std::size_t>> parse_ranks(const std::string& text) {
    std::vector<std::size_t> ranks;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty() ||
            item.find_first_not_of("0123456789") != std::string::npos ||
            item.size() > 6) {
            return std::nullopt;
        }
        ranks.push_back(std::stoul(item));
    }
    if (ranks.empty() || std::count(ranks.begin(), ranks.end(), 0U) != 0) {
        return std::nullopt;
    }
    return ranks;
}

/// Parse the arguments after the program name, or print why they are invalid.
std::optional<Options> parse_args(const std::vector<std::string>& args) {
    Options options;
    for (std::size_t index = 0; index + 1 < args.size(); index += 2) {
        const std::string& flag = args[index];
        const std::string& value = args[index + 1];
        const std::optional<std::vector<std::size_t>> ranks =
            flag == "--tp" ? parse_ranks(value) : std::nullopt;
        if (flag == "--model") {
            options.model_dir = value;
        } else if (flag == "--tokens") {
            options.tokens_file = value;
        } else if (flag == "--precision") {
            options.precision = value;
        } else if (flag == "--scheme") {
            options.scheme = value;
        } else if (flag == "--reference") {
            options.reference = value;
        } else if (ranks.has_value()) {
            options.ranks = *ranks;
        } else {
            std::cerr << kUsage;
            return std::nullopt;
        }
    }
    if (args.size() % 2 != 0 || options.model_dir.empty() ||
        options.tokens_file.empty() ||
        (options.precision != "fp32" && options.precision != "w8a8") ||
        (options.scheme != "megatron" && options.scheme != "column") ||
        (options.reference != "same" && options.reference != "fp32")) {
        std::cerr << kUsage;
        return std::nullopt;
    }
    return options;
}

/// Read one sequence of token IDs per non-empty line.
std::vector<TokenIds> read_sequences(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<TokenIds> sequences;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream stream(line);
        TokenIds ids;
        std::uint64_t id = 0;
        while (stream >> id) {
            ids.push_back(static_cast<std::uint32_t>(id));
        }
        if (!ids.empty()) {
            sequences.push_back(std::move(ids));
        }
    }
    return sequences;
}

/// Negative log-probability of @p target under @p logits, in double.
double nll(const Tensor<float>& logits, std::uint32_t target) {
    const float largest =
        *std::max_element(logits.values.begin(), logits.values.end());
    double sum = 0.0;
    for (const float value : logits.values) {
        sum += std::exp(static_cast<double>(value) - largest);
    }
    return std::log(sum) -
           (static_cast<double>(logits.values.at(target)) - largest);
}

/// Add one position's comparison of @p split against @p reference.
void accumulate(Totals& totals, const Tensor<float>& reference,
                const Tensor<float>& split,
                std::optional<std::uint32_t> target) {
    float largest = 0.0F;
    float worst = 0.0F;
    for (std::size_t i = 0; i < reference.values.size(); ++i) {
        largest = std::max(largest, std::fabs(reference.values[i]));
        worst =
            std::max(worst, std::fabs(reference.values[i] - split.values[i]));
    }
    const double rel = static_cast<double>(worst) / largest;
    totals.max_rel = std::max(totals.max_rel, rel);
    totals.sum_rel += rel;
    ++totals.positions;
    if (inference_engine::argmax(reference) ==
        inference_engine::argmax(split)) {
        ++totals.agree;
    }
    if (target) {
        totals.nll_reference += nll(reference, *target);
        totals.nll_split += nll(split, *target);
        ++totals.scored;
    }
}

/// Feed one sequence, teacher-forced, to the reference and every split
/// model, comparing them at every position.
void compare_sequence(GPTNeo<float>& reference,
                      std::vector<dist_inference::TpGptNeo<float>>& splits,
                      std::vector<Totals>& totals, const TokenIds& sequence) {
    for (std::size_t position = 0; position < sequence.size(); ++position) {
        const TokenIds step{sequence[position]};
        std::optional<std::uint32_t> target;
        if (position + 1 < sequence.size()) {
            target = sequence[position + 1];
        }
        const Tensor<float> expected = reference.forward(step);
        for (std::size_t index = 0; index < splits.size(); ++index) {
            accumulate(totals[index], expected, splits[index].forward(step),
                       target);
        }
    }
}

void print_row(const Options& options, const std::string& ranks,
               const Totals& totals) {
    const std::string& reference =
        options.reference == "fp32" ? options.reference : options.precision;
    std::cout << std::setprecision(6) << options.precision << '\t'
              << options.scheme << '\t' << ranks << '\t' << reference << '\t'
              << totals.positions << '\t' << totals.max_rel << '\t'
              << totals.sum_rel / static_cast<double>(totals.positions) << '\t'
              << static_cast<double>(totals.agree) /
                     static_cast<double>(totals.positions)
              << '\t'
              << std::exp(totals.nll_reference /
                          static_cast<double>(totals.scored))
              << '\t'
              << std::exp(totals.nll_split / static_cast<double>(totals.scored))
              << '\n'
              << std::flush;
}

} // namespace

int main(int argc, char** argv) {
    const std::optional<Options> options =
        parse_args(std::vector<std::string>(argv + 1, argv + argc));
    if (!options) {
        return 1;
    }

    try {
        const std::vector<TokenIds> sequences =
            read_sequences(options->tokens_file);
        const ModelWeights<float> fp32 =
            inference_engine::load_gpt_neo(options->model_dir);
        ModelWeights<float> weights = fp32;
        if (options->precision == "w8a8") {
            inference_engine::quantize_w8a8(weights);
        }
        const dist_inference::TpScheme scheme =
            options->scheme == "column" ? dist_inference::TpScheme::kColumnOnly
                                        : dist_inference::TpScheme::kMegatron;
        // With one rank, TpGptNeo is the single chip bit for bit.
        GPTNeo<float> reference(options->reference == "fp32" ? fp32 : weights);
        std::vector<Totals> totals(options->ranks.size());
        for (const TokenIds& sequence : sequences) {
            reference.reset();
            std::vector<dist_inference::TpGptNeo<float>> splits;
            for (const std::size_t ranks : options->ranks) {
                splits.emplace_back(weights, ranks, scheme);
            }
            compare_sequence(reference, splits, totals, sequence);
        }
        for (std::size_t index = 0; index < totals.size(); ++index) {
            print_row(options.value(), std::to_string(options->ranks[index]),
                      totals[index]);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
