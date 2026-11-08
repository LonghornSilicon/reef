// Runs GPT-Neo on N simulated Coral chips with Megatron tensor parallelism
// and checks it:
//
//   multi_sim --elf ELF --tp N [--precision fp32|w8a8] [--model DIR]
//             [--new-tokens N] [ID...]
//
// Each chip is a FuncSim (Spike, no timing model) running the tp_rank
// firmware on its rank blob. This program is the host: it embeds each token,
// sums the chips' partials in rank order at every sync, and gathers the
// logits (dist_coral/tp_host_if.hpp). Without --model it runs coral's small
// random test model; with it, a Hugging Face checkpoint, which needs as much
// host memory per chip as its share of the weights.
//
// The prompt (default 5 17 42 3 77) is fed a token at a time, then the
// chips' greedy tokens. Each token's logits are checked:
//   Test A: equal to dist_inference::TpGptNeo's, the same split on the host,
//           bit for bit. Any difference fails.
//   Test C: within Test B's tolerance of inference_engine::GPTNeo, one chip,
//           fed the same tokens: 1e-5 (fp32) or 5% (W8A8) of its largest
//           logit.
// Exits 0 if both pass, 1 if either fails or a chip traps.
#include "dist_coral/tp_blob.hpp"
#include "dist_coral/tp_host_if.hpp"
#include "dist_inference/shard.hpp"
#include "dist_inference/tp_model.hpp"
#include "pack_rank.hpp"

#include "common/MemoryMap.h"
#include "frontend/FuncSim.h"
#include "inference_engine/model.hpp"
#include "inference_engine/operator/embedding.hpp"
#include "inference_engine/weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using dist_coral::Phase;
using dist_coral::TpMailbox;
using inference_engine::ModelWeights;
using inference_engine::Tensor;
using inference_engine::TokenIds;

constexpr const char* kUsage =
    "usage: multi_sim --elf ELF --tp N [--precision fp32|w8a8] [--model DIR] "
    "[--new-tokens N] [ID...]\n";

// Where each chip's rank blob and arena live: memory off the core, past the
// highmem map's ITCM and DTCM. The functional model needs only that the
// firmware can address it.
constexpr std::uint32_t kExternalBase = 0x80000000;
constexpr std::uint32_t kAlign = 16;
// Instructions a chip may run between two pauses before it is called hung.
constexpr std::uint64_t kStepLimit = 2'000'000'000;

struct Options {
    std::string elf;
    std::size_t ranks = 0;
    std::string precision = "fp32";
    std::string model_dir;
    std::size_t new_tokens = 20;
    TokenIds prompt{5, 17, 42, 3, 77};
};

std::optional<Options> parse_args(const std::vector<std::string>& args) {
    Options options;
    TokenIds prompt;
    for (std::size_t index = 0; index < args.size(); ++index) {
        const std::string& arg = args[index];
        const bool has_value = index + 1 < args.size();
        if (arg == "--elf" && has_value) {
            options.elf = args[++index];
        } else if (arg == "--tp" && has_value) {
            options.ranks = std::stoul(args[++index]);
        } else if (arg == "--precision" && has_value) {
            options.precision = args[++index];
        } else if (arg == "--model" && has_value) {
            options.model_dir = args[++index];
        } else if (arg == "--new-tokens" && has_value) {
            options.new_tokens = std::stoul(args[++index]);
        } else if (!arg.empty() && arg[0] != '-') {
            prompt.push_back(static_cast<std::uint32_t>(std::stoul(arg)));
        } else {
            return std::nullopt;
        }
    }
    if (!prompt.empty()) {
        options.prompt = prompt;
    }
    if (options.elf.empty() || options.ranks == 0 ||
        (options.precision != "fp32" && options.precision != "w8a8")) {
        return std::nullopt;
    }
    return options;
}

/// coral's GPT-Neo test model (coral/host/gen_golden.cpp, small_config()):
/// global and local layers, a window shorter than the sequence.
ModelWeights<float> small_model() {
    inference_engine::GptNeoConfig config;
    config.vocab_size = 96;
    config.max_positions = 32;
    config.hidden_size = 32;
    config.num_layers = 4;
    config.num_heads = 4;
    config.inner_size = 128;
    config.window_size = 8;
    constexpr std::uint32_t kSeed = 7;
    return inference_engine::random_gpt_neo(config, kSeed);
}

std::uint32_t align(std::size_t bytes) {
    return static_cast<std::uint32_t>((bytes + kAlign - 1) / kAlign * kAlign);
}

/// One simulated chip and its view of the mailbox.
class Chip {
  public:
    Chip(const std::string& elf, const std::vector<std::uint8_t>& blob,
         std::size_t max_tokens, std::size_t rank)
        : rank_(rank), sim_(elf, options(blob, max_tokens)) {
        std::memcpy(memory(kExternalBase, blob.size()), blob.data(),
                    blob.size());
        TpMailbox& box = mailbox();
        box = {};
        box.model_address = kExternalBase;
        box.arena_address = kExternalBase + align(blob.size());
        box.arena_bytes = arena_bytes(blob, max_tokens);
        box.max_tokens = static_cast<std::uint32_t>(max_tokens);
        run();
        expect(Phase::kReady);
    }

    TpMailbox& mailbox() {
        return *reinterpret_cast<TpMailbox*>(
            memory(reef_kernels::kMailboxAddress, sizeof(TpMailbox)));
    }

    /// The exchange buffer, @p count floats of it.
    float* exchange(std::size_t count) {
        return reinterpret_cast<float*>(
            memory(mailbox().exchange_address, count * sizeof(float)));
    }

    /// Release the chip and run it to its next pause.
    void resume() {
        sim_.resume();
        run();
    }

    /// Throw unless the chip paused in @p phase.
    void expect(Phase phase) {
        TpMailbox& box = mailbox();
        if (box.phase != phase) {
            throw std::runtime_error(
                "rank " + std::to_string(rank_) + " paused in phase " +
                std::to_string(static_cast<unsigned>(box.phase)) +
                " with status " +
                std::to_string(static_cast<unsigned>(box.status)) +
                ", expected phase " +
                std::to_string(static_cast<unsigned>(phase)));
        }
    }

    [[nodiscard]] std::uint64_t instructions() const {
        return sim_.getNumExecuted();
    }

  private:
    static std::uint32_t arena_bytes(const std::vector<std::uint8_t>& blob,
                                     std::size_t max_tokens) {
        // Generous: every buffer rank_arena_bytes() counts, rounded up.
        dist_coral::RankHeader h{};
        std::memcpy(&h, blob.data(), sizeof h);
        const std::size_t width =
            static_cast<std::size_t>(h.local_heads) * h.head_dim;
        const std::size_t floats =
            (2 * h.num_layers * max_tokens * width) + (2 * h.hidden_size) +
            (4 * width) + h.local_inner + max_tokens +
            std::max<std::size_t>(h.hidden_size, h.local_vocab);
        constexpr std::size_t kSlack = 4096;
        return align((floats * sizeof(float)) +
                     std::max<std::size_t>(h.hidden_size, h.local_inner) +
                     kSlack);
    }

    static reef_perf::SpikeOptions
    options(const std::vector<std::uint8_t>& blob, std::size_t max_tokens) {
        reef_perf::SpikeOptions spike;
        spike.Regions = reef_perf::getHighmemMemoryMap();
        spike.Regions.push_back(
            {kExternalBase, align(blob.size()) + arena_bytes(blob, max_tokens),
             reef_perf::RegionKind::External});
        return spike;
    }

    char* memory(std::uint32_t address, std::size_t bytes) {
        char* host =
            sim_.getHostMemory(address, static_cast<std::uint32_t>(bytes));
        if (host == nullptr) {
            throw std::runtime_error("rank " + std::to_string(rank_) +
                                     ": unmapped address " +
                                     std::to_string(address));
        }
        return host;
    }

    void run() {
        const std::uint64_t start = sim_.getNumExecuted();
        while (sim_.next()) {
            if (sim_.getNumExecuted() - start > kStepLimit) {
                throw std::runtime_error("rank " + std::to_string(rank_) +
                                         " did not pause");
            }
        }
    }

    std::size_t rank_;
    reef_perf::FuncSim sim_;
};

/// The host side of the protocol: N chips, each paused in kReady.
class Host {
  public:
    Host(const ModelWeights<float>& weights, const Options& options,
         std::size_t max_tokens)
        : weights_(weights) {
        const auto precision = options.precision == "w8a8"
                                   ? reef_kernels::Precision::kW8A8
                                   : reef_kernels::Precision::kFloat32;
        for (std::size_t rank = 0; rank < options.ranks; ++rank) {
            chips_.push_back(std::make_unique<Chip>(
                options.elf,
                dist_coral_host::pack_rank(weights, precision, options.ranks,
                                           rank),
                max_tokens, rank));
        }
    }

    /// Run @p token on every chip; with @p want_logits, return the logits.
    Tensor<float> step(std::uint32_t token, bool want_logits) {
        const std::size_t hidden = weights_.config.hidden_size;
        const Tensor<float> x = inference_engine::add_position_embeddings(
            inference_engine::lookup_token_embeddings({token}, weights_.wte),
            weights_.wpe, position_++);
        for (auto& chip : chips_) {
            std::memcpy(chip->exchange(hidden), x.values.data(),
                        hidden * sizeof(float));
            chip->mailbox().command = dist_coral::Command::kStep;
            chip->mailbox().want_logits = want_logits ? 1 : 0;
            chip->resume();
        }
        for (const auto& layer : weights_.layers) {
            combine(layer.attention.output_bias);
            combine(layer.mlp.c_proj_bias);
        }
        if (!want_logits) {
            expect_all(Phase::kReady);
            return {};
        }
        expect_all(Phase::kLogits);
        std::vector<Tensor<float>> parts;
        for (std::size_t rank = 0; rank < chips_.size(); ++rank) {
            const std::size_t count =
                dist_inference::shard_range(weights_.config.vocab_size,
                                            chips_.size(), rank)
                    .count;
            const float* logits = chips_[rank]->exchange(count);
            parts.push_back({{1, count}, {logits, logits + count}});
        }
        return dist_inference::concat_columns(parts);
    }

    void stop() {
        for (auto& chip : chips_) {
            chip->mailbox().command = dist_coral::Command::kStop;
            chip->resume();
            chip->expect(Phase::kDone);
        }
    }

    [[nodiscard]] std::uint64_t instructions() const {
        std::uint64_t total = 0;
        for (const auto& chip : chips_) {
            total += chip->instructions();
        }
        return total;
    }

  private:
    /// One sum sync: add the partials in rank order, then the bias, and
    /// hand the sum back to every chip.
    void combine(const Tensor<float>& bias) {
        const std::size_t hidden = weights_.config.hidden_size;
        expect_all(Phase::kPartial);
        std::vector<Tensor<float>> partials;
        for (auto& chip : chips_) {
            const float* partial = chip->exchange(hidden);
            partials.push_back({{1, hidden}, {partial, partial + hidden}});
        }
        const Tensor<float> sum = dist_inference::sum_in_rank_order(
            partials, dist_inference::OptionalBias<float>(std::cref(bias)));
        for (auto& chip : chips_) {
            std::memcpy(chip->exchange(hidden), sum.values.data(),
                        hidden * sizeof(float));
            chip->resume();
        }
    }

    void expect_all(Phase phase) {
        for (auto& chip : chips_) {
            chip->expect(phase);
        }
    }

    const ModelWeights<float>& weights_;
    std::vector<std::unique_ptr<Chip>> chips_;
    std::size_t position_ = 0;
};

/// Index of the first value whose bits differ, or nothing.
std::optional<std::size_t> first_difference(const Tensor<float>& got,
                                            const Tensor<float>& want) {
    for (std::size_t i = 0; i < want.values.size(); ++i) {
        std::uint32_t a = 0;
        std::uint32_t b = 0;
        std::memcpy(&a, &got.values.at(i), sizeof a);
        std::memcpy(&b, &want.values[i], sizeof b);
        if (a != b) {
            return i;
        }
    }
    return std::nullopt;
}

/// Largest logit difference over the largest reference logit.
double relative_error(const Tensor<float>& got, const Tensor<float>& want) {
    float largest = 0.0F;
    float worst = 0.0F;
    for (std::size_t i = 0; i < want.values.size(); ++i) {
        largest = std::max(largest, std::fabs(want.values[i]));
        worst = std::max(worst, std::fabs(want.values[i] - got.values[i]));
    }
    return static_cast<double>(worst) / largest;
}

int run(const Options& options) {
    ModelWeights<float> weights =
        options.model_dir.empty()
            ? small_model()
            : inference_engine::load_gpt_neo(options.model_dir);
    const bool w8a8 = options.precision == "w8a8";
    if (w8a8) {
        inference_engine::quantize_w8a8(weights);
    }
    const double tolerance = w8a8 ? 5e-2 : 1e-5;
    const std::size_t max_tokens = options.prompt.size() + options.new_tokens;

    Host host(weights, options, max_tokens);
    dist_inference::TpGptNeo<float> split(weights, options.ranks,
                                          dist_inference::TpScheme::kMegatron);
    inference_engine::GPTNeo<float> single(weights);

    TokenIds fed = options.prompt;
    TokenIds generated;
    bool passed = true;
    double worst_c = 0.0;
    for (std::size_t i = 0; i < max_tokens - 1; ++i) {
        const bool want_logits = i + 1 >= options.prompt.size();
        const Tensor<float> chips = host.step(fed[i], want_logits);
        const Tensor<float> expected = split.forward({fed[i]});
        const Tensor<float> one_chip = single.forward({fed[i]});
        if (!want_logits) {
            continue;
        }
        if (const auto index = first_difference(chips, expected)) {
            std::cout << "FAIL test A: token " << generated.size() << ", logit "
                      << *index << ": chips " << chips.values[*index]
                      << ", TpGptNeo " << expected.values[*index] << '\n';
            passed = false;
            break;
        }
        const double error = relative_error(chips, one_chip);
        worst_c = std::max(worst_c, error);
        if (error > tolerance) {
            std::cout << "FAIL test C: token " << generated.size()
                      << ", relative error " << error << " > " << tolerance
                      << '\n';
            passed = false;
        }
        generated.push_back(
            static_cast<std::uint32_t>(inference_engine::argmax(chips)));
        fed.push_back(generated.back());
    }
    if (passed) {
        host.stop();
    }

    std::cout << options.precision << " megatron tp=" << options.ranks << ": "
              << generated.size() << " tokens";
    for (const std::uint32_t token : generated) {
        std::cout << ' ' << token;
    }
    std::cout << "\n  test A (bitwise vs TpGptNeo): "
              << (passed ? "pass" : "see above")
              << "\n  test C (vs one chip): max relative error " << worst_c
              << " (tolerance " << tolerance << ") over " << generated.size()
              << " tokens"
              << "\n  instructions, all chips: " << host.instructions() << '\n';
    return passed ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    const std::optional<Options> options =
        parse_args(std::vector<std::string>(argv + 1, argv + argc));
    if (!options) {
        std::cerr << kUsage;
        return 2;
    }
    try {
        return run(*options);
    } catch (const std::exception& error) {
        std::cerr << "multi_sim: " << error.what() << '\n';
        return 1;
    }
}
