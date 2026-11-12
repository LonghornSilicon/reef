#pragma once

// How the host drives N chips running one GPT-Neo with Megatron tensor
// parallelism. Each chip runs the same firmware on its own rank blob
// (tp_blob.hpp). Every sync is an mpause: the chip writes its output to the
// exchange buffer, says what it holds in phase, and halts; the host combines
// all chips' outputs, writes the result back, and releases them.
//
//   1. The host loads each chip's rank blob, fills its mailbox (model, arena,
//      max_tokens) and releases it. The chip binds the model and pauses in
//      kReady, with exchange_address set, or kDone with an error status.
//   2. For each token the host writes the token's embedding, wte[id] +
//      wpe[position], to every chip's exchange buffer, sets command to
//      kStep and want_logits, and releases the chips.
//   3. Twice per layer, after the attention output projection and after
//      c_proj, every chip pauses in kPartial with a partial [hidden] in the
//      exchange buffer. The host sums them in rank order, rank 0 first, adds
//      the bias once (dist_inference::sum_in_rank_order), writes the sum to
//      every chip and releases them. Each chip adds it to its residual
//      stream.
//   4. At the end of the step each chip pauses in kLogits, its lm_head rows
//      [local_vocab] in the exchange buffer, if want_logits was set, and in
//      kReady otherwise. The host concatenates the logits in rank order.
//   5. Command kStop makes the chip set kDone and halt for good.
//
// So a step takes 2 * layers + 1 syncs with logits, as TpGptNeo::syncs()
// records. The linker script places the mailbox first in DTCM, at
// reef_kernels::kMailboxAddress.

#include "reef_kernels/host_if.hpp"

#include <cstdint>

namespace dist_coral {

enum class Command : std::uint32_t {
    kStep = 0, // run one token, its embedding in the exchange buffer
    kStop = 1, // halt
};

enum class Phase : std::uint32_t {
    kStarting = 0, // not yet paused
    kReady = 1,    // waiting for a command
    kPartial = 2,  // exchange holds a partial [hidden] to sum
    kLogits = 3,   // exchange holds this rank's logits [local_vocab]
    kDone = 4,     // halted for good; see status
};

struct TpMailbox {
    // In, before the first release.
    std::uint32_t model_address; // this chip's rank blob
    std::uint32_t arena_address; // scratch for the firmware
    std::uint32_t arena_bytes;
    std::uint32_t max_tokens; // positions the KV cache must hold
    // In, with each command.
    Command command;
    std::uint32_t want_logits;
    // Out, at each pause.
    Phase phase;
    reef_kernels::Status status;
    std::uint32_t exchange_address; // float [max(hidden, local_vocab)]
};

} // namespace dist_coral
