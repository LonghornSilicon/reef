//===- Inst.h - The instruction object --------------------------*- C++ -*-===//
///
/// \file
/// The instruction object that flows through the timing model. The frontend
/// creates one per dynamic instruction, already executed by Spike; units pass
/// InstRef over Sparta ports and fill in the timing fields. Every module can
/// read every field; docs/interfaces.md lists which module writes which.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_COMMON_INST_H
#define REEF_PERF_COMMON_INST_H

#include <cstdint>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace reef_perf {

/// Coarse instruction classes; each maps to an execution resource and to
/// latency/occupancy parameters.
enum class InstClass : uint8_t {
  ALU,         ///< Integer ALU, Zbb, lui/auipc.
  BRANCH,      ///< Conditional branches.
  JUMP,        ///< jal, jalr.
  MUL,         ///< mul*.
  DIV,         ///< div*, rem*.
  CSR,         ///< csrr*.
  FENCE,       ///< fence, fence.i.
  SYSTEM,      ///< ecall, ebreak, mret, wfi, mpause.
  LOAD,        ///< Scalar integer loads.
  STORE,       ///< Scalar integer stores.
  FP,          ///< FP32 add/mul/fma/compare/convert/move.
  FP_DIV,      ///< fdiv.s, fsqrt.s.
  FP_LOAD,     ///< flw.
  FP_STORE,    ///< fsw.
  VSET,        ///< vsetvli, vsetivli, vsetvl.
  V_ALU,       ///< Vector integer ALU, mask ops, moves.
  V_MUL,       ///< Vector multiply and multiply-accumulate.
  V_DIV,       ///< Vector integer divide.
  V_FP,        ///< Vector FP32 arithmetic.
  V_FDIV,      ///< Vector FP divide and square root.
  V_PERM,      ///< Reductions, slides, gathers, compress.
  V_LOAD,      ///< Vector loads.
  V_STORE,     ///< Vector stores.
  V_TO_SCALAR, ///< Vector ops that write a scalar or FP register.
  MATRIX,      ///< Matrix-engine ops (opaque; nothing decodes to it yet).
  UNKNOWN,     ///< Not recognized; timed as an ALU op.
  NUM_CLASSES  ///< Number of classes (not a class).
};

/// \returns the short lower-case name of \p Class, e.g. "alu" or "v_load".
const char *getClassName(InstClass Class);

// Scoreboard ids: one numbering across the three register files.
constexpr uint16_t XBase = 0;    ///< Scoreboard id of x0.
constexpr uint16_t FBase = 32;   ///< Scoreboard id of f0.
constexpr uint16_t VBase = 64;   ///< Scoreboard id of v0.
constexpr uint16_t NumRegs = 96; ///< Number of scoreboard ids.

/// One memory access performed by an instruction.
struct MemAccess {
  uint32_t Addr = 0;    ///< Byte address.
  uint8_t Size = 0;     ///< Size in bytes.
  bool IsStore = false; ///< True for a store.
};

/// One dynamic instruction and its progress through the timing model.
struct Inst {
  /// Marks a timing field that has not been set yet.
  static constexpr uint64_t Never = std::numeric_limits<uint64_t>::max();

  uint64_t Seq = 0;           ///< 0-based dynamic instruction number.
  uint32_t PC = 0;            ///< Address of the instruction.
  uint32_t NextPC = 0;        ///< Address executed next (branch outcome).
  uint32_t Encoding = 0;      ///< Raw 32-bit instruction word.
  std::string Disasm;         ///< Spike's disassembly.
  std::string Mnemonic;       ///< First token of Disasm, e.g. "addi".
  std::vector<MemAccess> Mem; ///< Accesses; one per element for vectors.
  uint32_t VL = 0;            ///< Vector length when it executed.
  uint8_t SEWBytes = 1;       ///< Selected element width in bytes.
  uint8_t LMUL8 = 8;          ///< LMUL x 8 (LMUL=1 is 8, 1/2 is 4).
  InstClass Class = InstClass::UNKNOWN; ///< Coarse class.
  std::vector<uint16_t> Srcs;           ///< Registers read (no x0).
  std::vector<uint16_t> Dsts;           ///< Registers written (no x0).

  uint64_t FetchCycle = 0;           ///< Cycle it was fetched.
  uint64_t DispatchCycle = 0;        ///< Cycle it was dispatched.
  uint64_t IssueCycle = 0;           ///< Cycle its execution started.
  uint64_t ResultReadyCycle = Never; ///< First cycle a dependent may dispatch.
  uint64_t CompleteCycle = Never;    ///< First cycle it may retire.
  uint64_t RetireCycle = 0;          ///< Cycle it retired.

  /// \returns true for a taken branch, a jump or a trap.
  bool isRedirect() const { return NextPC != PC + 4; }

  /// \returns true for VSET through V_TO_SCALAR.
  bool isVector() const {
    return Class >= InstClass::VSET && Class <= InstClass::V_TO_SCALAR;
  }

  /// \returns true for MATRIX.
  bool isMatrix() const { return Class == InstClass::MATRIX; }

  /// \returns true for scalar, FP and vector loads and stores.
  bool isMemory() const {
    return Class == InstClass::LOAD || Class == InstClass::STORE ||
           Class == InstClass::FP_LOAD || Class == InstClass::FP_STORE ||
           Class == InstClass::V_LOAD || Class == InstClass::V_STORE;
  }
};

/// Shared handle to an instruction; this is what Sparta ports carry.
using InstRef = std::shared_ptr<Inst>;

/// Prints \p I to \p OS (for Sparta's port logging). \returns \p OS.
std::ostream &operator<<(std::ostream &OS, const Inst &I);

} // namespace reef_perf

#endif // REEF_PERF_COMMON_INST_H
