#include "tp_rank.hpp"

#include "reef_kernels/kernels.hpp"

namespace firmware {
namespace {

using dist_coral::Phase;
using dist_coral::RankHeader;
using dist_coral::RankLayer;
using reef_kernels::BlobLinear;
using reef_kernels::Precision;
using reef_kernels::Status;

constexpr std::size_t kAlign = 16;

std::size_t aligned(std::size_t bytes) {
    return (bytes + kAlign - 1) & ~(kAlign - 1);
}

std::size_t larger(std::size_t a, std::size_t b) { return a > b ? a : b; }

// Every buffer the firmware carves from the arena, in order.
struct Layout {
    std::size_t cache_floats, hidden, width, inner, scores, exchange, x_q;

    Layout(const RankHeader& h, std::size_t max_tokens)
        : cache_floats(static_cast<std::size_t>(h.num_layers) * max_tokens *
                       h.local_heads * h.head_dim),
          hidden(h.hidden_size),
          width(static_cast<std::size_t>(h.local_heads) * h.head_dim),
          inner(h.local_inner), scores(max_tokens),
          exchange(larger(h.hidden_size, h.local_vocab)),
          x_q(larger(h.hidden_size, h.local_inner)) {}

    [[nodiscard]] std::size_t bytes() const {
        constexpr std::size_t kHiddenRows = 2;
        constexpr std::size_t kWidthRows = 4;
        return (2 * aligned(cache_floats * sizeof(float))) +
               (kHiddenRows * aligned(hidden * sizeof(float))) +
               (kWidthRows * aligned(width * sizeof(float))) +
               aligned(inner * sizeof(float)) +
               aligned(scores * sizeof(float)) +
               aligned(exchange * sizeof(float)) + aligned(x_q);
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

// The mailbox serve() answers; the sync callback needs it.
dist_coral::TpMailbox* active = nullptr;

// Stops until the host releases this chip. The memory clobber keeps every
// store before the pause and every load of the host's reply after it.
void pause() { asm volatile(".word 0x08000073" ::: "memory"); }

void sync_with_host(Phase phase) {
    active->phase = phase;
    pause();
}

} // namespace

std::size_t rank_arena_bytes(const RankHeader& header, std::size_t max_tokens) {
    return Layout(header, max_tokens).bytes();
}

TpRank::TpRank(const std::uint8_t* blob, std::size_t max_tokens,
               std::uint8_t* arena, std::size_t arena_size)
    : blob_(blob), header_(reinterpret_cast<const RankHeader*>(blob)),
      layers_(nullptr), max_tokens_(max_tokens) {
    const RankHeader& h = *header_;
    if (h.magic != dist_coral::kRankBlobMagic ||
        h.version != dist_coral::kRankBlobVersion || h.local_heads == 0 ||
        h.head_dim == 0) {
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
    layers_ = at<RankLayer>(h.layers);

    Carver carve(arena);
    key_cache_ = carve.take<float>(layout.cache_floats);
    value_cache_ = carve.take<float>(layout.cache_floats);
    x_ = carve.take<float>(layout.hidden);
    normed_ = carve.take<float>(layout.hidden);
    query_ = carve.take<float>(layout.width);
    key_ = carve.take<float>(layout.width);
    value_ = carve.take<float>(layout.width);
    context_ = carve.take<float>(layout.width);
    inner_ = carve.take<float>(layout.inner);
    scores_ = carve.take<float>(layout.scores);
    exchange_ = carve.take<float>(layout.exchange);
    x_q_ = carve.take<std::int8_t>(layout.x_q);
}

void TpRank::project(const BlobLinear& weight, std::size_t in, std::size_t out,
                     const float* x, float* y) {
    if (header_->precision == Precision::kW8A8) {
        reef_kernels::linear_w8a8(1, in, out, x, at<std::int8_t>(weight.weight),
                                  at<float>(weight.scales), x_q_, y);
    } else {
        reef_kernels::linear_f32(1, in, out, x, at<float>(weight.weight), y);
    }
}

void TpRank::project_bias(const BlobLinear& weight, std::uint32_t bias,
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

void TpRank::step(bool want_logits, Sync sync) {
    const RankHeader& h = *header_;
    const std::size_t hidden = h.hidden_size;
    const std::size_t heads = h.local_heads;
    const std::size_t head_dim = h.head_dim;
    const std::size_t width = heads * head_dim;
    const std::size_t position = position_++;

    // The host embedded the token: x = wte[token] + wpe[position].
    for (std::size_t i = 0; i < hidden; ++i) {
        x_[i] = exchange_[i];
    }

    for (std::size_t l = 0; l < h.num_layers; ++l) {
        const RankLayer& layer = layers_[l];

        // x = x + sum over ranks of attention partials, + bias (host)
        reef_kernels::layernorm_f32(1, hidden, x_, at<float>(layer.ln_1_weight),
                                    at<float>(layer.ln_1_bias),
                                    h.layer_norm_eps, normed_);
        project(layer.query, hidden, width, normed_, query_);
        project(layer.key, hidden, width, normed_, key_);
        project(layer.value, hidden, width, normed_, value_);
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
        project(layer.output, width, hidden, context_, exchange_);
        sync(Phase::kPartial);
        reef_kernels::add_f32(hidden, x_, exchange_, x_);

        // x = x + sum over ranks of c_proj(gelu(c_fc(ln_2(x)))) partials
        reef_kernels::layernorm_f32(1, hidden, x_, at<float>(layer.ln_2_weight),
                                    at<float>(layer.ln_2_bias),
                                    h.layer_norm_eps, normed_);
        project_bias(layer.fc, layer.fc_bias, hidden, h.local_inner, normed_,
                     inner_);
        reef_kernels::gelu_f32(h.local_inner, inner_, inner_);
        project(layer.proj, h.local_inner, hidden, inner_, exchange_);
        sync(Phase::kPartial);
        reef_kernels::add_f32(hidden, x_, exchange_, x_);
    }

    if (want_logits) {
        // This rank's rows of logits = ln_f(x) . wte^T, the tied lm_head
        reef_kernels::layernorm_f32(1, hidden, x_, at<float>(h.ln_f_weight),
                                    at<float>(h.ln_f_bias), h.layer_norm_eps,
                                    normed_);
        reef_kernels::matvec_rows_f32(h.local_vocab, hidden,
                                      at<float>(h.lm_head), normed_, exchange_);
    }
}

void serve(dist_coral::TpMailbox& mailbox) {
    active = &mailbox;
    TpRank rank(reinterpret_cast<const std::uint8_t*>(
                    static_cast<std::uintptr_t>(mailbox.model_address)),
                mailbox.max_tokens,
                reinterpret_cast<std::uint8_t*>(
                    static_cast<std::uintptr_t>(mailbox.arena_address)),
                mailbox.arena_bytes);
    if (!rank.ok()) {
        mailbox.status = rank.error();
        mailbox.phase = Phase::kDone;
        return;
    }
    mailbox.exchange_address = static_cast<std::uint32_t>(
        reinterpret_cast<std::uintptr_t>(rank.exchange()));
    std::size_t steps = 0;
    sync_with_host(Phase::kReady);
    while (mailbox.command == dist_coral::Command::kStep) {
        if (steps++ == mailbox.max_tokens) {
            mailbox.status = Status::kTooLong;
            break;
        }
        const bool want_logits = mailbox.want_logits != 0;
        rank.step(want_logits, sync_with_host);
        sync_with_host(want_logits ? Phase::kLogits : Phase::kReady);
    }
    if (mailbox.status == Status::kIdle) {
        mailbox.status = Status::kDone;
    }
    mailbox.phase = Phase::kDone;
}

} // namespace firmware
