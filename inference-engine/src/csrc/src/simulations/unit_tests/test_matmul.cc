// Checks every matmul variant (RVV and VME, signed and unsigned A, and the
// matmul() dispatcher) against a plain scalar reference on random inputs,
// including shapes that are not multiples of 16. Cycle counts come from the
// sim benches in sim/, not from here.
//
// Host build, from simulations/ (emulates both RVV and VME):
//   clang++ -std=c++17 -DVME_EMULATE -DRVV_EMULATE -Ikernels -Iincludes \
//       kernels/matmul.cc kernels/matmul_rvv.cc kernels/vme_matmul.cc \
//       unit_tests/test_matmul.cc -o test_matmul
#include "matmul.h"
#include "matmul_rvv.h"
#include "vme_matmul.h"

#include <stddef.h>
#include <stdint.h>

extern "C" int printf(const char*, ...);

namespace {

constexpr size_t kMaxA = 32 * 768;
constexpr size_t kMaxB = 768 * 64;
constexpr size_t kMaxC = 32 * 64;

// too big for the stack together; main() holds one as a static local
struct Buffers {
    uint8_t a[kMaxA];
    uint8_t b[kMaxB];
    int32_t c[kMaxC];
    int32_t ref[kMaxC];
};

uint32_t next_random(uint32_t* state) {
    *state = (*state * 1664525U) + 1013904223U;
    return *state >> 16;
}

void fill(Buffers* buf, size_t n, size_t k, size_t m) {
    auto state = static_cast<uint32_t>((n * 131) + (k * 7) + m);
    for (size_t i = 0; i < n * k; ++i) {
        buf->a[i] = static_cast<uint8_t>(next_random(&state));
    }
    for (size_t i = 0; i < k * m; ++i) {
        buf->b[i] = static_cast<uint8_t>(next_random(&state));
    }
}

void reference(Buffers* buf, size_t n, size_t k, size_t m, bool signed_a) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            int32_t acc = 0;
            for (size_t t = 0; t < k; ++t) {
                const uint8_t raw = buf->a[(i * k) + t];
                const int32_t a = signed_a ? static_cast<int8_t>(raw) : raw;
                acc += a * static_cast<int32_t>(buf->b[(t * m) + j]);
            }
            buf->ref[(i * m) + j] = acc;
        }
    }
}

// Runs one variant on the current inputs and compares it with the reference.
template <typename Run>
int check(Buffers* buf, const char* variant, size_t n, size_t k, size_t m,
          bool signed_a, Run run) {
    for (size_t i = 0; i < n * m; ++i) {
        buf->c[i] = 0x5A5A5A5A; // poison, so unwritten outputs fail
    }
    reference(buf, n, k, m, signed_a);
    run();
    size_t bad = 0;
    for (size_t i = 0; i < n * m; ++i) {
        bad += buf->c[i] != buf->ref[i] ? 1 : 0;
    }
    printf("%-16s %3zux%3zux%3zu %s", variant, n, k, m,
           bad == 0 ? "pass\n" : "FAIL");
    if (bad != 0) {
        printf(" (%zu of %zu wrong)\n", bad, n * m);
    }
    return bad == 0 ? 0 : 1;
}

int check_shape(Buffers* buf, size_t n, size_t k, size_t m) {
    fill(buf, n, k, m);
    const uint8_t* a = &buf->a[0];
    const uint8_t* b = &buf->b[0];
    int32_t* c = &buf->c[0];
    const auto* a_s8 = reinterpret_cast<const int8_t*>(a);
    auto* c_u32 = reinterpret_cast<uint32_t*>(c);
    int failures = 0;
    failures += check(buf, "matmul_rvv", n, k, m, false,
                      [&] { matmul_rvv(n, k, m, a, b, c_u32); });
    failures += check(buf, "matmul_rvv_s8u8", n, k, m, true,
                      [&] { matmul_rvv_s8u8(n, k, m, a_s8, b, c); });
    if (n % 16 == 0 && m % 16 == 0 && k % 4 == 0) {
        failures += check(buf, "vme_matmul_u8", n, k, m, false,
                          [&] { vme_matmul_u8(n, k, m, a, b, c); });
        failures += check(buf, "vme_matmul_s8u8", n, k, m, true,
                          [&] { vme_matmul_s8u8(n, k, m, a_s8, b, c); });
    }
    failures += check(buf, "matmul (u8)", n, k, m, false,
                      [&] { matmul(n, k, m, a, false, b, c); });
    failures += check(buf, "matmul (s8)", n, k, m, true,
                      [&] { matmul(n, k, m, a, true, b, c); });
    return failures;
}

} // namespace

int main() {
    static Buffers buf;
    int failures = 0;
    failures += check_shape(&buf, 1, 768, 64); // decode: q . kt, x . W tile
    failures += check_shape(&buf, 7, 13, 23);  // nothing a multiple of 16
    failures += check_shape(&buf, 3, 5, 40);   // m > 16: several RVV strips
    failures += check_shape(&buf, 16, 4, 16);  // one VME step
    failures += check_shape(&buf, 32, 8, 48);
    failures += check_shape(&buf, 16, 768, 64); // prefill tile
    failures += check_shape(&buf, 32, 64, 64);
    if (failures != 0) {
        printf("%d check(s) failed\n", failures);
    }
    return failures;
}
