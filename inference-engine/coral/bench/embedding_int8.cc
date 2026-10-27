// int8 embedding of 16 tokens of width 64 from a 64-entry vocabulary.

#include "fill.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
constexpr std::size_t kTokens = 16;
constexpr std::size_t kWidth = 64;
constexpr std::size_t kVocab = 64;
std::uint32_t ids[kTokens];
std::int8_t wte[kVocab * kWidth];
std::int8_t wpe[kTokens * kWidth];
std::int8_t y[kTokens * kWidth];
} // namespace

int main() {
    bench::Fill fill;
    fill.integers(ids, kTokens, 26); // reduced into the vocabulary below
    for (std::uint32_t& id : ids) {
        id %= kVocab;
    }
    fill.int8s(wte, kVocab * kWidth);
    fill.int8s(wpe, kTokens * kWidth);
    for (std::size_t t = 0; t < kTokens; ++t) {
        reef_kernels::embedding_int8(kWidth, ids[t],
                                     static_cast<std::uint32_t>(t), wte, wpe,
                                     y + (t * kWidth));
    }
    return 0;
}
