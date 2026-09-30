// C interface to the CoralNPU MPACT instruction-set simulator, used by the
// perf model for execute-at-fetch.
//
// The perf model calls cn_step() once per instruction it fetches. MPACT
// executes that instruction immediately and fills in a cn_inst_t describing
// it (PC, encoding, disassembly, memory addresses touched, vector config).
//
// This is a plain C ABI on purpose: MPACT is built with Bazel/clang and the
// perf model with CMake/g++, so the two sides share nothing but this header.

#ifndef CORALNPU_PERF_DRIVER_H_
#define CORALNPU_PERF_DRIVER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CN_API __attribute__((visibility("default")))

// Enough for a segment load/store with LMUL=8, SEW=8, NF=8 at VLEN=128.
#define CN_MAX_MEM 1024
#define CN_DISASM_LEN 96

typedef struct {
  uint32_t addr;
  uint8_t size;      // bytes
  uint8_t is_store;  // 0 = load, 1 = store
} cn_mem_t;

typedef struct {
  uint64_t seq;       // 0-based dynamic instruction number
  uint32_t pc;
  uint32_t next_pc;   // PC of the next instruction executed (branch outcome)
  uint32_t encoding;  // raw 32-bit instruction word
  char disasm[CN_DISASM_LEN];

  // Vector configuration in effect when this instruction executed.
  uint32_t vl;
  uint8_t sew_bytes;  // 1, 2 or 4
  uint8_t lmul8;      // LMUL * 8 (so LMUL=1 -> 8, LMUL=1/2 -> 4)

  uint16_t num_mem;       // valid entries in mem[]
  uint8_t mem_truncated;  // 1 if more than CN_MAX_MEM accesses happened
  cn_mem_t mem[CN_MAX_MEM];
} cn_inst_t;

typedef struct {
  uint32_t itcm_start;
  uint32_t itcm_length;
  uint32_t dtcm_start;
  uint32_t dtcm_length;
} cn_options_t;

// Default options match the M3 default memory map (8 KB ITCM, 32 KB DTCM).
CN_API void cn_default_options(cn_options_t* opts);

// Returns NULL on failure (message printed to stderr).
CN_API void* cn_create(const cn_options_t* opts, const char* elf_path);

// Executes exactly one instruction.
// Returns 1 if an instruction executed and *out is valid,
//         0 if the program has halted (mpause/ebreak/exit),
//        -1 on error (message printed to stderr).
CN_API int cn_step(void* handle, cn_inst_t* out);

CN_API void cn_destroy(void* handle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // CORALNPU_PERF_DRIVER_H_
