// The one matmul entry point: picks the matrix engine when the shape fits
// its 16 x 16 x 4 steps, the vector unit otherwise. See matmul.h.
#include "matmul.h"

#include "matmul_rvv.h"
#include "vme_helpers.h"
#include "vme_matmul.h"

extern "C" void matmul(size_t n, size_t k, size_t m, const uint8_t* a,
                       bool signed_a, const uint8_t* b, int32_t* c) {
    const auto* a_s8 = reinterpret_cast<const int8_t*>(a);
    const bool fits_vme = n % vme::kTileDim == 0 && m % vme::kTileDim == 0 &&
                          k % vme::kKPerStep == 0;
    if (fits_vme && signed_a) {
        vme_matmul_s8u8(n, k, m, a_s8, b, c);
    } else if (fits_vme) {
        vme_matmul_u8(n, k, m, a, b, c);
    } else if (signed_a) {
        matmul_rvv_s8u8(n, k, m, a_s8, b, c);
    } else {
        matmul_rvv(n, k, m, a, b, reinterpret_cast<uint32_t*>(c));
    }
}
