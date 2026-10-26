// The firmware's whole forward pass against GPTNeo (model.hpp): the logits
// of every generated token, then the same request through the host mailbox
// exactly as firmware/main.cc serves it. Runs once in fp32 and once in W8A8.

#include "check.hpp"
#include "golden/gpt_neo.hpp"
#include "gpt_neo.hpp"
#include "reef_kernels/host_if.hpp"

#include <cstdint>

namespace {

constexpr std::size_t kArenaBytes = 64 * 1024;
alignas(16) std::uint8_t arena[kArenaBytes];
reef_kernels::Mailbox mailbox;

std::uint32_t address(const void* pointer) {
    return static_cast<std::uint32_t>(
        reinterpret_cast<std::uintptr_t>(pointer));
}

} // namespace

int main() {
    std::size_t which = 0;
    for (const golden::ModelCase& c : golden::kGptNeo) {
        const auto* blob = reinterpret_cast<const std::uint8_t*>(c.blob);

        firmware::GptNeo model(blob, golden::kMaxTokens, arena, kArenaBytes);
        if (!model.ok()) {
            check::fail(which, static_cast<std::size_t>(model.error()));
        }
        std::uint32_t next = 0;
        for (std::size_t i = 0; i < c.prompt_length; ++i) {
            next = model.step(c.prompt[i], i + 1 == c.prompt_length);
        }
        for (std::size_t g = 0; g < c.new_tokens; ++g) {
            check::expect_bits(which, model.logits(),
                               c.logits + (g * c.vocab_size), c.vocab_size);
            check::expect_equal(which, next, c.tokens[g]);
            if (g + 1 < c.new_tokens) {
                next = model.step(next, true);
            }
        }
        ++which;

        mailbox = {};
        mailbox.model_address = address(blob);
        mailbox.arena_address = address(arena);
        mailbox.arena_bytes = kArenaBytes;
        mailbox.prompt_length = static_cast<std::uint32_t>(c.prompt_length);
        mailbox.new_tokens = static_cast<std::uint32_t>(c.new_tokens);
        for (std::size_t i = 0; i < c.prompt_length; ++i) {
            mailbox.tokens[i] = c.prompt[i];
        }
        firmware::serve(mailbox);
        check::expect_equal(
            which, static_cast<std::uint32_t>(mailbox.status),
            static_cast<std::uint32_t>(reef_kernels::Status::kDone));
        for (std::size_t g = 0; g < c.new_tokens; ++g) {
            check::expect_equal(which, mailbox.tokens[c.prompt_length + g],
                                c.tokens[g]);
        }
        ++which;
    }
    return 0;
}
