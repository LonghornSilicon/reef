#pragma once

// How the host drives the firmware. The core has no console and no driver
// stack, so the protocol is one request per run:
//
//   1. The host writes the packed model (model_blob.hpp) somewhere the core
//      can read, and fills the mailbox below: where the model is, a scratch
//      arena for the KV cache and activations, and the prompt.
//   2. The host releases the core from reset. crt0 runs main, which serves
//      the request (firmware/gpt_neo.hpp, serve()).
//   3. The firmware appends the generated tokens after the prompt, sets
//      status, and halts with mpause. The host sees the core halt and reads
//      the mailbox back.
//
// The linker script places the mailbox first in DTCM, at kMailboxAddress.

#include <cstdint>

namespace reef_kernels {

// The first byte of DTCM in the memory map the program was linked for.
#ifdef REEF_KERNELS_M3_MAP
inline constexpr std::uint32_t kMailboxAddress = 0x00010000; // link_m3.ld
#else
inline constexpr std::uint32_t kMailboxAddress = 0x00100000; // link.ld
#endif
inline constexpr std::uint32_t kMailboxTokens = 256;

enum class Status : std::uint32_t {
    kIdle = 0,
    kDone = 1,
    kBadModel = 2, // the blob's magic or version is wrong
    kTooLong = 3,  // prompt + new tokens exceed the model or the mailbox
    kNoMemory = 4, // the arena is smaller than the request needs
};

struct Mailbox {
    std::uint32_t model_address; // packed model
    std::uint32_t arena_address; // scratch for the firmware
    std::uint32_t arena_bytes;
    std::uint32_t prompt_length;
    std::uint32_t new_tokens;
    Status status;
    // In: the prompt. Out: the prompt followed by the generated tokens.
    std::uint32_t tokens[kMailboxTokens];
};

} // namespace reef_kernels
