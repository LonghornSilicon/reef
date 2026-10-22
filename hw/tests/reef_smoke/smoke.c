// Reef smoke test program: scalar int, scalar FP, RVV int32 and RVV FP32 (zve32f).
// Results land in smoke_result (DTCM); the testbench reads them back over SPI.
#include <riscv_vector.h>
#include <stdint.h>

#define N 64

struct result {
  uint32_t magic;       // 0x600DCAFE on success, 0xBAD00BAD after a trap
  uint32_t vadd_sum;    // sum of (a+b) computed with vector adds
  uint32_t fsum_bits;   // bits of the FP32 vector reduction
  uint32_t trap_cause;  // mcause if a trap happened
  uint32_t scalar_mul;  // scalar integer check
  uint32_t scalar_fp;   // bits of a scalar FP result
  uint32_t vl_max;      // vsetvlmax e32m1 (4 at VLEN 128)
  uint32_t cycles;      // mcycle delta over main's body
};

volatile struct result smoke_result __attribute__((aligned(16)));

static int32_t a[N], b[N], c[N];
static float fa[N];

int main(void) {
  uint32_t t0;
  __asm__ volatile("csrr %0, mcycle" : "=r"(t0));
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = 3 * i + 1;
    fa[i] = (float)i * 0.5f;
  }

  // Vector int32 add, strip-mined.
  for (size_t i = 0, vl; i < N; i += vl) {
    vl = __riscv_vsetvl_e32m1(N - i);
    vint32m1_t va = __riscv_vle32_v_i32m1(&a[i], vl);
    vint32m1_t vb = __riscv_vle32_v_i32m1(&b[i], vl);
    __riscv_vse32_v_i32m1(&c[i], __riscv_vadd_vv_i32m1(va, vb, vl), vl);
  }
  uint32_t sum = 0;
  for (int i = 0; i < N; i++) sum += (uint32_t)c[i];

  // Vector FP32: sum(fa[i] * 2.0f) = sum(i) = 2016.
  vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  for (size_t i = 0, vl; i < N; i += vl) {
    vl = __riscv_vsetvl_e32m1(N - i);
    vfloat32m1_t v = __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(&fa[i], vl), 2.0f, vl);
    acc = __riscv_vfredusum_vs_f32m1_f32m1(v, acc, vl);
  }
  float fsum = __riscv_vfmv_f_s_f32m1_f32(acc);

  volatile int32_t x = 1234, y = -567;
  volatile float p = 1.5f, q = 2.25f;
  float pq = p * q + 0.125f;  // 3.5

  union { float f; uint32_t u; } u1 = {fsum}, u2 = {pq};
  smoke_result.vadd_sum = sum;
  smoke_result.fsum_bits = u1.u;
  smoke_result.scalar_mul = (uint32_t)(x * y);
  smoke_result.scalar_fp = u2.u;
  smoke_result.vl_max = (uint32_t)__riscv_vsetvlmax_e32m1();
#ifdef TEST_BUS_ERROR
  // Off-chip address: reef_axi_err_slave answers DECERR -> load access fault (mcause 5).
  volatile uint32_t sink = *(volatile uint32_t *)0x80000000u;
  (void)sink;
#endif
  uint32_t t1;
  __asm__ volatile("csrr %0, mcycle" : "=r"(t1));
  smoke_result.cycles = t1 - t0;
  smoke_result.magic = 0x600DCAFE;
  return 0;
}
