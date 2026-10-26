#pragma once

// Benchmarks fill their inputs on the core from a fixed seed, then run one
// kernel once. reef_perf counts the whole program, fill included; use
// reef-perf's reef_trace to isolate the kernel's instructions.
//
// Float and W8A8 shapes are TinyStories-Instruct-8M decode (one token):
// hidden 256, MLP inner 1024, 16 heads of 16. The *_int8 benchmarks keep
// the shapes of the original standalone int8 kernels (16 rows of 64), so
// their numbers compare with the earlier runs.

#include <cstddef>
#include <cstdint>

namespace bench {

inline constexpr std::size_t kHidden = 256;
inline constexpr std::size_t kInner = 1024;
inline constexpr std::size_t kHeadDim = 16;

class Fill {
  public:
    // Uniform in [-scale, scale).
    void floats(float* x, std::size_t n, float scale) {
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = scale * (static_cast<float>(next()) * kUnit - 1.0F);
        }
    }

    void int8s(std::int8_t* x, std::size_t n) { integers(x, n, 24); }

    // Any integer type, from the top bits of the generator: shift 24 gives
    // a full-range byte, 16 a 16-bit value, 20 a value within +-2048 when
    // T is signed 32-bit and the result is centred.
    template <typename T> void integers(T* x, std::size_t n, unsigned shift) {
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = static_cast<T>(static_cast<std::int32_t>(next()) >>
                                  static_cast<std::int32_t>(shift));
        }
    }

  private:
    static constexpr float kUnit = 1.0F / 2147483648.0F; // 2^-31

    std::uint32_t next() {
        state_ = (state_ * 1664525U) + 1013904223U;
        return state_;
    }

    std::uint32_t state_ = 1;
};

} // namespace bench
