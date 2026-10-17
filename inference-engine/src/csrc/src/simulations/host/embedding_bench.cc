// Host-side (x86/arm64) GPT-2 embedding: token lookup plus learned position
// embedding, requantized to int8 in the [seq x d_model] row-major layout the
// first layernorm reads. Times decode (1 token) and every prefill length in
// config/model.json with a steady clock, and prints JSON for run_all.py.
//
// Weights are random int8 tables of GPT-2's shape (wte 50257 x 768, wpe
// 1024 x 768): the cost depends on the shapes, not the values.
//
// Host build, from simulations/:
//   clang++ -std=c++17 -O2 -Iincludes -I../../include \
//       host/embedding_bench.cc -o embedding_bench
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "inference_engine/operator/embedding.hpp"
#include "params.h"

namespace {

using inference_engine::Tensor;
using inference_engine::TokenIds;

constexpr int kRuns = 1000;
constexpr double kTargetMs = 20.0;

int8_t to_int8(float value) {
    const float rounded = value >= 0 ? value + 0.5F : value - 0.5F;
    return static_cast<int8_t>(std::clamp(rounded, -128.0F, 127.0F));
}

// Token lookup (reused from the inference engine), then position add with
// requantization from the shared embedding scale to the activation scale.
std::vector<int8_t> embed(const TokenIds& ids, std::size_t position_offset,
                          const Tensor<int8_t>& wte,
                          const Tensor<int8_t>& wpe) {
    const Tensor<int8_t> tokens =
        inference_engine::lookup_token_embeddings(ids, wte);
    const std::size_t width = params::kDModel;
    const float scale = params::kEmbeddingScale / params::kActScale;
    std::vector<int8_t> out(tokens.values.size());
    for (std::size_t t = 0; t < ids.size(); ++t) {
        const int8_t* pos = &wpe.values[(position_offset + t) * width];
        for (std::size_t c = 0; c < width; ++c) {
            const std::size_t i = (t * width) + c;
            const int sum = tokens.values[i] + pos[c];
            out[i] = to_int8(static_cast<float>(sum) * scale);
        }
    }
    return out;
}

Tensor<int8_t> random_table(std::size_t rows, std::mt19937* rng) {
    std::uniform_int_distribution<int> dist(-128, 127);
    Tensor<int8_t> table{{rows, params::kDModel},
                         std::vector<int8_t>(rows * params::kDModel)};
    for (int8_t& value : table.values) {
        value = static_cast<int8_t>(dist(*rng));
    }
    return table;
}

// Times kRuns calls of embed() for one sequence length; prints a JSON row.
void time_length(const char* phase, std::size_t length,
                 std::size_t position_offset, const Tensor<int8_t>& wte,
                 const Tensor<int8_t>& wpe, std::mt19937* rng, bool last) {
    std::uniform_int_distribution<std::uint32_t> token(0, params::kVocab - 1);
    TokenIds ids(length);
    for (std::uint32_t& id : ids) {
        id = token(*rng);
    }
    std::vector<double> ms(kRuns);
    std::size_t sink = 0; // keeps the work from being optimized away
    for (double& sample : ms) {
        const auto start = std::chrono::steady_clock::now();
        sink += static_cast<std::uint8_t>(
            embed(ids, position_offset, wte, wpe).back());
        const auto stop = std::chrono::steady_clock::now();
        sample =
            std::chrono::duration<double, std::milli>(stop - start).count();
    }
    std::sort(ms.begin(), ms.end());
    const double median = ms[kRuns / 2];
    const double p95 = ms[(kRuns * 95) / 100];
    std::printf("  {\"phase\": \"%s\", \"tokens\": %zu, \"median_ms\": %.6f, "
                "\"p95_ms\": %.6f, \"runs\": %d, \"target_ms\": %.1f, "
                "\"pass\": %s, \"sink\": %zu}%s\n",
                phase, length, median, p95, kRuns, kTargetMs,
                p95 < kTargetMs ? "true" : "false", sink, last ? "" : ",");
}

} // namespace

int main() {
    std::mt19937 rng(42);
    const Tensor<int8_t> wte = random_table(params::kVocab, &rng);
    const Tensor<int8_t> wpe = random_table(params::kNCtx, &rng);
    std::printf("[\n");
    // decode: one token at the last position of the decode context
    time_length("decode", 1, params::kDecodeContext - 1, wte, wpe, &rng, false);
    const std::size_t count = std::size(params::kPrefillLens);
    for (std::size_t i = 0; i < count; ++i) {
        time_length("prefill", params::kPrefillLens[i], 0, wte, wpe, &rng,
                    i + 1 == count);
    }
    std::printf("]\n");
    return 0;
}
