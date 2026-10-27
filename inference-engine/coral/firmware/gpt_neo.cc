#include "gpt_neo.hpp"

#include "reef_kernels/kernels.hpp"

namespace firmware {
namespace {

using reef_kernels::BlobHeader;
using reef_kernels::BlobLayer;
using reef_kernels::BlobLinear;
using reef_kernels::Precision;
using reef_kernels::Status;

constexpr std::size_t kAlign = 16;

std::size_t aligned(std::size_t bytes) {
    return (bytes + kAlign - 1) & ~(kAlign - 1);
}

// Every buffer the firmware carves from the arena, in order.
struct Layout {
    std::size_t cache_floats, hidden, inner, scores, logits, x_q;

    Layout(const BlobHeader& h, std::size_t max_tokens)
        : cache_floats(static_cast<std::size_t>(h.num_layers) * max_tokens *
                       h.hidden_size),
          hidden(h.hidden_size), inner(h.inner_size), scores(max_tokens),
          logits(h.vocab_size),
          x_q(h.inner_size > h.hidden_size ? h.inner_size : h.hidden_size) {}

    [[nodiscard]] std::size_t bytes() const {
        constexpr std::size_t kHiddenRows = 7;
        return (2 * aligned(cache_floats * sizeof(float))) +
               (kHiddenRows * aligned(hidden * sizeof(float))) +
               aligned(inner * sizeof(float)) +
               aligned(scores * sizeof(float)) +
               aligned(logits * sizeof(float)) + aligned(x_q);
    }
};

// Hands out aligned pieces of the arena; sizes were checked up front.
class Carver {
  public:
    explicit Carver(std::uint8_t* base) : next_(base) {}

    template <typename T> T* take(std::size_t count) {
        T* piece = reinterpret_cast<T*>(next_);
        next_ += aligned(count * sizeof(T));
        return piece;
    }

  private:
    std::uint8_t* next_;
};

} // namespace

std::size_t arena_bytes(const BlobHeader& header, std::size_t max_tokens) {
    return Layout(header, max_tokens).bytes();
}

GptNeo::GptNeo(const std::uint8_t* blob, std::size_t max_tokens,
               std::uint8_t* arena, std::size_t arena_size)
    : blob_(blob), header_(reinterpret_cast<const BlobHeader*>(blob)),
      layers_(nullptr), max_tokens_(max_tokens) {
    const BlobHeader& h = *header_;
    if (h.magic != reef_kernels::kBlobMagic ||
        h.version != reef_kernels::kBlobVersion || h.num_heads == 0 ||
        h.hidden_size % h.num_heads != 0) {
        error_ = Status::kBadModel;
        return;
    }
    if (max_tokens == 0 || max_tokens > h.max_positions) {
        error_ = Status::kTooLong;
        return;
    }
    const Layout layout(h, max_tokens);
    if (layout.bytes() > arena_size) {
        error_ = Status::kNoMemory;
        return;
    }
    layers_ = at<BlobLayer>(h.layers);

    Carver carve(arena);
    key_cache_ = carve.take<float>(layout.cache_floats);
    value_cache_ = carve.take<float>(layout.cache_floats);
    x_ = carve.take<float>(layout.hidden);
    normed_ = carve.take<float>(layout.hidden);
    query_ = carve.take<float>(layout.hidden);
    key_ = carve.take<float>(layout.hidden);
    value_ = carve.take<float>(layout.hidden);
    context_ = carve.take<float>(layout.hidden);
    sublayer_ = carve.take<float>(layout.hidden);
    inner_ = carve.take<float>(layout.inner);
    scores_ = carve.take<float>(layout.scores);
    logits_ = carve.take<float>(layout.logits);
    x_q_ = carve.take<std::int8_t>(layout.x_q);
}

void GptNeo::project(const BlobLinear& weight, std::size_t in, std::size_t out,
                     const float* x, float* y) {
    if (header_->precision == Precision::kW8A8) {
        reef_kernels::linear_w8a8(1, in, out, x, at<std::int8_t>(weight.weight),
                                  at<float>(weight.scales), x_q_, y);
    } else {
        reef_kernels::linear_f32(1, in, out, x, at<float>(weight.weight), y);
    }
}

void GptNeo::project_bias(const BlobLinear& weight, std::uint32_t bias,
                          std::size_t in, std::size_t out, const float* x,
                          float* y) {
    if (header_->precision == Precision::kW8A8) {
        reef_kernels::linear_w8a8_bias(
            1, in, out, x, at<std::int8_t>(weight.weight),
            at<float>(weight.scales), at<float>(bias), x_q_, y);
    } else {
        reef_kernels::linear_bias_f32(1, in, out, x, at<float>(weight.weight),
                                      at<float>(bias), y);
    }
}

std::uint32_t GptNeo::step(std::uint32_t token, bool want_logits) {
    const BlobHeader& h = *header_;
    const std::size_t hidden = h.hidden_size;
    const std::size_t heads = h.num_heads;
    const std::size_t head_dim = hidden / heads;
    const std::size_t position = position_++;

    // x = wte[token] + wpe[position]
    reef_kernels::embedding_f32(hidden, token,
                                static_cast<std::uint32_t>(position),
                                at<float>(h.wte), at<float>(h.wpe), x_);

    for (std::size_t l = 0; l < h.num_layers; ++l) {
        const BlobLayer& layer = layers_[l];

        // x = x + attention(ln_1(x))
        reef_kernels::layernorm_f32(1, hidden, x_, at<float>(layer.ln_1_weight),
                                    at<float>(layer.ln_1_bias),
                                    h.layer_norm_eps, normed_);
        project(layer.query, hidden, hidden, normed_, query_);
        project(layer.key, hidden, hidden, normed_, key_);
        project(layer.value, hidden, hidden, normed_, value_);
        for (std::size_t head = 0; head < heads; ++head) {
            // This head's cache: [max_tokens][head_dim] rows.
            const std::size_t base =
                ((l * heads) + head) * max_tokens_ * head_dim;
            float* keys = key_cache_ + base;
            float* values = value_cache_ + base;
            for (std::size_t d = 0; d < head_dim; ++d) {
                keys[(position * head_dim) + d] = key_[(head * head_dim) + d];
                values[(position * head_dim) + d] =
                    value_[(head * head_dim) + d];
            }
            reef_kernels::attention_head_f32(
                1, position + 1, head_dim, position,
                layer.is_local != 0 ? h.window_size : 0,
                query_ + (head * head_dim), head_dim, keys, values, scores_,
                context_ + (head * head_dim), head_dim);
        }
        project_bias(layer.output, layer.output_bias, hidden, hidden, context_,
                     sublayer_);
        reef_kernels::add_f32(hidden, x_, sublayer_, x_);

        // x = x + mlp(ln_2(x)), mlp = c_proj(gelu(c_fc(.)))
        reef_kernels::layernorm_f32(1, hidden, x_, at<float>(layer.ln_2_weight),
                                    at<float>(layer.ln_2_bias),
                                    h.layer_norm_eps, normed_);
        project_bias(layer.fc, layer.fc_bias, hidden, h.inner_size, normed_,
                     inner_);
        reef_kernels::gelu_f32(h.inner_size, inner_, inner_);
        project_bias(layer.proj, layer.proj_bias, h.inner_size, hidden, inner_,
                     sublayer_);
        reef_kernels::add_f32(hidden, x_, sublayer_, x_);
    }

    if (!want_logits) {
        return 0;
    }
    // logits = ln_f(x) . wte^T, the tied lm_head
    reef_kernels::layernorm_f32(1, hidden, x_, at<float>(h.ln_f_weight),
                                at<float>(h.ln_f_bias), h.layer_norm_eps,
                                normed_);
    reef_kernels::matvec_rows_f32(h.vocab_size, hidden, at<float>(h.wte),
                                  normed_, logits_);
    return reef_kernels::argmax_f32(h.vocab_size, logits_);
}

void serve(reef_kernels::Mailbox& mailbox) {
    const std::size_t prompt = mailbox.prompt_length;
    const std::size_t total = prompt + mailbox.new_tokens;
    if (prompt == 0 || total > reef_kernels::kMailboxTokens) {
        mailbox.status = Status::kTooLong;
        return;
    }
    GptNeo model(reinterpret_cast<const std::uint8_t*>(
                     static_cast<std::uintptr_t>(mailbox.model_address)),
                 total,
                 reinterpret_cast<std::uint8_t*>(
                     static_cast<std::uintptr_t>(mailbox.arena_address)),
                 mailbox.arena_bytes);
    if (!model.ok()) {
        mailbox.status = model.error();
        return;
    }
    // Greedy decoding as GPTNeo::generate: the last prompt token's logits
    // pick the first new token, and each new token picks the next.
    std::uint32_t next = 0;
    for (std::size_t i = 0; i < prompt; ++i) {
        next = model.step(mailbox.tokens[i], i + 1 == prompt);
    }
    for (std::size_t g = 0; g < mailbox.new_tokens; ++g) {
        mailbox.tokens[prompt + g] = next;
        if (g + 1 < mailbox.new_tokens) {
            next = model.step(next, true);
        }
    }
    mailbox.status = Status::kDone;
}

} // namespace firmware
