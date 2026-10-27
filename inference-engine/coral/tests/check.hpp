#pragma once

// Pass/fail for golden tests on the core. A test that returns from main
// passes: crt0 then runs mpause and reef_perf exits 0. A mismatch runs
// ebreak, which reef_perf reports as a trap and exits non-zero. Before
// trapping, the case and element that differed are left in a0 and a1 for
// reef_trace or a debugger.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace check {

[[noreturn]] inline void fail(std::size_t which_case, std::size_t element) {
    asm volatile("mv a0, %0\n\tmv a1, %1\n\tebreak" ::"r"(which_case),
                 "r"(element)
                 : "a0", "a1");
    __builtin_unreachable();
}

// Floats must match bit for bit, including the sign of zero.
inline void expect_bits(std::size_t which_case, const float* got,
                        const float* want, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t a = 0;
        std::uint32_t b = 0;
        std::memcpy(&a, got + i, sizeof a);
        std::memcpy(&b, want + i, sizeof b);
        if (a != b) {
            fail(which_case, i);
        }
    }
}

inline void expect_equal(std::size_t which_case, std::uint32_t got,
                         std::uint32_t want) {
    if (got != want) {
        fail(which_case, got);
    }
}

} // namespace check
