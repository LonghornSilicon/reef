// The embedding operator (operator/embedding.hpp) for Coral NPU: token row
// lookup from a streamed slice of the table, and the position add. See
// kernels.h.
#include "helpers.h"
#include "kernels.h"

extern "C" size_t embedding_lookup(size_t length, size_t width,
                                   const uint32_t* ids, uint32_t vocab_start,
                                   size_t vocab_rows, const int8_t* table,
                                   int8_t* y) {
    size_t filled = 0;
    for (size_t position = 0; position < length; ++position) {
        const uint32_t id = ids[position];
        if (id < vocab_start || id - vocab_start >= vocab_rows) {
            continue; // in another slice of the table
        }
        const int8_t* row =
            table + (static_cast<size_t>(id - vocab_start) * width);
        int8_t* out = y + (position * width);
        for (size_t c = 0; c < width; ++c) {
            out[c] = row[c];
        }
        ++filled;
    }
    return filled;
}

extern "C" void add_position_embeddings(size_t length, size_t width,
                                        const int8_t* tokens, float token_scale,
                                        const int8_t* position_table,
                                        float position_scale,
                                        size_t position_offset, float out_scale,
                                        int8_t* y) {
    const int8_t* positions = position_table + (position_offset * width);
    for (size_t i = 0; i < length * width; ++i) {
        const float value = (token_scale * static_cast<float>(tokens[i])) +
                            (position_scale * static_cast<float>(positions[i]));
        y[i] = quant::to_int8(value, out_scale);
    }
}
