// Residual add, embedding and argmax against residual_add(),
// lookup_token_embeddings() + add_position_embeddings(), and argmax().

#include "check.hpp"
#include "golden/elementwise.hpp"
#include "reef_kernels/kernels.hpp"

namespace {
float out[golden::kScratchFloats];
}

int main() {
    std::size_t which = 0;
    for (const golden::AddCase& c : golden::kAdd) {
        reef_kernels::add_f32(c.n, c.a, c.b, out);
        check::expect_bits(which++, out, c.y, c.n);
    }
    for (const golden::EmbeddingCase& c : golden::kEmbedding) {
        reef_kernels::embedding_f32(c.width, c.id, c.position, c.wte, c.wpe,
                                    out);
        check::expect_bits(which++, out, c.y, c.width);
    }
    for (const golden::ArgmaxCase& c : golden::kArgmax) {
        check::expect_equal(which++, reef_kernels::argmax_f32(c.n, c.x),
                            c.index);
    }
    return 0;
}
