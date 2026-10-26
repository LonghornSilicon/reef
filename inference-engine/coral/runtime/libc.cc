// The four functions GCC may call even in a freestanding program, for
// struct copies and zeroing loops. Built with
// -fno-tree-loop-distribute-patterns so GCC cannot turn these loops back
// into calls to themselves.

#include <cstddef>

extern "C" {

void* memset(void* dst, int value, std::size_t n) {
    auto* d = static_cast<unsigned char*>(dst);
    for (std::size_t i = 0; i < n; ++i) {
        d[i] = static_cast<unsigned char>(value);
    }
    return dst;
}

void* memcpy(void* dst, const void* src, std::size_t n) {
    auto* d = static_cast<unsigned char*>(dst);
    const auto* s = static_cast<const unsigned char*>(src);
    for (std::size_t i = 0; i < n; ++i) {
        d[i] = s[i];
    }
    return dst;
}

void* memmove(void* dst, const void* src, std::size_t n) {
    auto* d = static_cast<unsigned char*>(dst);
    const auto* s = static_cast<const unsigned char*>(src);
    if (d < s) {
        for (std::size_t i = 0; i < n; ++i) {
            d[i] = s[i];
        }
    } else {
        for (std::size_t i = n; i > 0; --i) {
            d[i - 1] = s[i - 1];
        }
    }
    return dst;
}

int memcmp(const void* a, const void* b, std::size_t n) {
    const auto* x = static_cast<const unsigned char*>(a);
    const auto* y = static_cast<const unsigned char*>(b);
    for (std::size_t i = 0; i < n; ++i) {
        if (x[i] != y[i]) {
            return x[i] < y[i] ? -1 : 1;
        }
    }
    return 0;
}

} // extern "C"
