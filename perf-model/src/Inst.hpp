// The instruction object that flows through the model.
//
// One Inst is created per dynamic instruction, at fetch, from the record that
// MPACT produced when it executed the instruction (execute-at-fetch). Units
// pass InstPtr (a shared pointer) to each other over Sparta ports and fill in
// the timing fields as the instruction moves down the pipeline.

#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace coralnpu_perf {

// Coarse instruction classes. Each class maps to one execution resource in
// Execute (see Execute.cpp) and to a latency/occupancy parameter.
enum class InstClass : uint8_t {
  ALU,          // integer ALU, Zbb, lui/auipc
  BRANCH,       // conditional branches
  JUMP,         // jal, jalr
  MUL,          // mul*
  DIV,          // div*, rem*
  CSR,          // csrr*
  FENCE,        // fence, fence.i
  SYSTEM,       // ecall, ebreak, mret, wfi, mpause
  LOAD,         // scalar integer loads
  STORE,        // scalar integer stores
  FP,           // FP32 add/mul/fma/compare/convert/move
  FP_DIV,       // fdiv.s, fsqrt.s
  FP_LOAD,      // flw
  FP_STORE,     // fsw
  VSET,         // vsetvli, vsetivli, vsetvl
  V_ALU,        // vector integer ALU, mask ops, moves
  V_MUL,        // vector multiply / multiply-accumulate
  V_DIV,        // vector integer divide
  V_FP,         // vector FP32 arithmetic
  V_FDIV,       // vector FP divide / sqrt
  V_PERM,       // reductions, slides, gathers, compress
  V_LOAD,       // vector loads
  V_STORE,      // vector stores
  V_TO_SCALAR,  // vector ops that write a scalar/FP register (vmv.x.s, vcpop, ...)
  UNKNOWN,
  NUM_CLASSES
};

const char* className(InstClass c);

// Register ids used by the scoreboard: one flat space for all register files,
// the same numbering the CoralNPU RTL uses for its retirement buffer.
constexpr uint16_t kXBase = 0;    // x0..x31
constexpr uint16_t kFBase = 32;   // f0..f31
constexpr uint16_t kVBase = 64;   // v0..v31
constexpr uint16_t kNumRegs = 96;

struct MemAccess {
  uint32_t addr;
  uint8_t size;
  bool is_store;
};

struct Inst {
  // --- Filled at fetch from the MPACT record ---
  uint64_t seq = 0;
  uint32_t pc = 0;
  uint32_t next_pc = 0;
  uint32_t encoding = 0;
  std::string disasm;
  std::string mnemonic;
  std::vector<MemAccess> mem;
  uint32_t vl = 0;         // vector length in effect
  uint8_t sew_bytes = 1;   // selected element width, bytes
  uint8_t lmul8 = 8;       // LMUL * 8

  // --- Filled by InstDecode ---
  InstClass cls = InstClass::UNKNOWN;
  std::vector<uint16_t> srcs;  // registers read (flat ids, x0 excluded)
  std::vector<uint16_t> dsts;  // registers written (flat ids, x0 excluded)

  // --- Timing, filled as the instruction moves through the model ---
  static constexpr uint64_t kNever = std::numeric_limits<uint64_t>::max();
  uint64_t fetch_cycle = 0;
  uint64_t dispatch_cycle = 0;
  uint64_t issue_cycle = 0;              // cycle it started in its unit
  uint64_t result_ready_cycle = kNever;  // first cycle a dependent may dispatch
  uint64_t complete_cycle = kNever;      // cycle it may retire
  uint64_t retire_cycle = 0;

  // True if the next instruction is not at pc+4 (taken branch, jump, trap).
  bool redirected() const { return next_pc != pc + 4; }

  bool isVector() const {
    return cls >= InstClass::VSET && cls <= InstClass::V_TO_SCALAR;
  }
  bool isMemory() const {
    return cls == InstClass::LOAD || cls == InstClass::STORE ||
           cls == InstClass::FP_LOAD || cls == InstClass::FP_STORE ||
           cls == InstClass::V_LOAD || cls == InstClass::V_STORE;
  }
};

using InstPtr = std::shared_ptr<Inst>;

// What Fetch sends to Dispatch each fetch cycle.
struct FetchPacket {
  std::vector<InstPtr> insts;
  bool last = false;  // true once the program has halted; no more packets follow
};

// Printers used by Sparta when it logs port traffic.
inline std::ostream& operator<<(std::ostream& os, const Inst& inst) {
  return os << "#" << inst.seq << " 0x" << std::hex << inst.pc << std::dec << " "
            << inst.disasm;
}
inline std::ostream& operator<<(std::ostream& os, const FetchPacket& pkt) {
  os << "[" << pkt.insts.size() << " insts" << (pkt.last ? ", last" : "") << "]";
  return os;
}

}  // namespace coralnpu_perf
