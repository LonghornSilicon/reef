// Prints the dynamic instruction stream of an ELF as MPACT executes it.
//
//   cn_trace <elf> [max_instructions]
//
// One line per instruction: seq, pc, encoding, next_pc, vl/sew/lmul, number
// of memory accesses, and the disassembly.

#include <cstdio>
#include <cstdlib>

#include "perf_driver/coralnpu_driver.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <elf> [max_instructions]\n", argv[0]);
    return 2;
  }
  const unsigned long long max_insts =
      argc > 2 ? std::strtoull(argv[2], nullptr, 0) : ~0ull;

  void* h = cn_create(nullptr, argv[1]);
  if (h == nullptr) return 1;

  static cn_inst_t rec;
  unsigned long long n = 0;
  int rc = 0;
  while (n < max_insts && (rc = cn_step(h, &rec)) == 1) {
    std::printf("%6llu %08x %08x -> %08x vl=%u e%u m%u/8 mem=%u  %s\n",
                static_cast<unsigned long long>(rec.seq), rec.pc, rec.encoding,
                rec.next_pc, rec.vl, rec.sew_bytes * 8, rec.lmul8,
                rec.num_mem, rec.disasm);
    ++n;
  }
  std::printf("# %llu instructions, %s\n", n,
              rc == 0 ? "halted" : (rc < 0 ? "error" : "limit reached"));
  cn_destroy(h);
  return rc < 0 ? 1 : 0;
}
