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

#include "common/Types.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <ostream>
#include <span>
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
  MATRIX,      ///< Matrix-engine (Zvt) ops other than mset*; see Inst::Mtx.
  UNKNOWN,     ///< Not recognized; timed as an ALU op.
  NUM_CLASSES  ///< Number of classes (not a class).
};

/// \returns the short lower-case name of \p Class, e.g. "alu" or "v_load".
const char *getClassName(InstClass Class);

/// One memory access performed by an instruction.
struct MemAccess {
  Address Addr = 0;     ///< Byte address.
  uint8_t Size = 0;     ///< Size in bytes.
  bool IsStore = false; ///< True for a store.
};

/// Bytes in one vector register (VLEN = 128 on M3).
inline constexpr uint32_t VectorRegisterBytes = 16;

/// What a MATRIX instruction does to the matrix engine (docs/matrix.md).
/// The Zvt mset* instructions are not here: they are vector configuration
/// (InstClass::VSET), as on CoralNPU, where the RVV front end runs them.
enum class MatrixOp : uint8_t {
  None,         ///< Not a matrix instruction.
  Zero,         ///< vtzero: clear a tile group.
  Discard,      ///< vtdiscard: mark tile state dead (mstatus.MS).
  MoveToTile,   ///< vtmv.t.v: vector registers to one tile slice.
  MoveToVector, ///< vtmv.v.t: one tile slice to vector registers.
  TileLoad,     ///< vtle8/16/32: memory to one tile slice.
  TileStore,    ///< vtse8/16/32: one tile slice to memory.
  Mac,          ///< vtmmu, vtmms, vtfmm: tile multiply-accumulate.
  /// A tile instruction the RVV decoder rejects (vl = 0, bad shape or
  /// alignment): it retires as a no-op without reaching the engine.
  Nop,
  NumOps ///< Number of ops (not an op).
};

/// \returns the short name of \p Op, e.g. "mac" or "tile_load".
const char *getMatrixOpName(MatrixOp Op);

/// Element type of a matrix multiply-accumulate's inputs.
enum class MatrixType : uint8_t {
  Int8, ///< int8 x int8 -> int32 (vtmmu, vtmms).
  Bf16, ///< bf16 x bf16 -> fp32 (vtfmm at SEW16).
  Fp32  ///< fp32 x fp32 -> fp32 (vtfmm at SEW32).
};

/// Matrix-engine details of a MATRIX instruction, filled in by the frontend
/// from the encoding and the Zvt configuration (mtype, vl) it ran under.
struct MatrixInfo {
  MatrixOp Op = MatrixOp::None; ///< What the instruction does.
  uint8_t FirstTile = 0; ///< First architectural tile touched (mt0-mt15).
  /// Tiles touched: 1, 2 or 4 (8-, 16- or 32-bit tile elements group
  /// adjacent tiles, zvt_ctrl.sv).
  uint8_t NumTiles = 1;
  uint8_t TM = 0; ///< Rows of C (mtype.tm), for Mac.
  /// Columns of C for Mac (vl); elements moved for loads, stores and moves.
  uint8_t TN = 0;
  uint8_t TK = 0;                     ///< Multiply depth (mtype.tk), for Mac.
  MatrixType Type = MatrixType::Int8; ///< Input type, for Mac.
  /// Data bytes per element: the memory element for loads and stores
  /// (EEW), SEW for moves.
  uint8_t ElemBytes = 1;
};

/// Splits a vector memory instruction's \p Accesses (one per element) into
/// one group per vector register of \p RegBytes bytes, in element order.
/// Spike records only active elements, so a masked access groups them as if
/// they were contiguous. \throws std::invalid_argument if \p RegBytes is 0.
std::vector<std::span<const MemAccess>>
splitByRegister(std::span<const MemAccess> Accesses, uint32_t RegBytes);

/// One dynamic instruction and its progress through the timing model.
struct Inst {
  /// Marks a timing field that has not been set yet.
  static constexpr Cycle Never = std::numeric_limits<Cycle>::max();

  uint64_t Seq = 0;      ///< 0-based dynamic instruction number.
  Address PC = 0;        ///< Address of the instruction.
  Address NextPC = 0;    ///< Address executed next (branch outcome).
  uint32_t Encoding = 0; ///< Raw 32-bit instruction word.
  uint32_t Rs1Val =
      0; ///< Value of integer source register rs1 before execution.
  uint32_t Rs2Val =
      0; ///< Value of integer source register rs2 before execution.
  std::string Disasm;         ///< Spike's disassembly.
  std::string Mnemonic;       ///< First token of Disasm, e.g. "addi".
  std::vector<MemAccess> Mem; ///< Accesses; one per element for vectors.
  uint32_t VL = 0;            ///< Vector length when it executed.
  uint8_t SEWBytes = 1;       ///< Selected element width in bytes.
  uint8_t LMUL8 = 8;          ///< LMUL x 8 (LMUL=1 is 8, 1/2 is 4).
  InstClass Class = InstClass::UNKNOWN; ///< Coarse class.
  std::vector<RegId> Srcs;              ///< Registers read (no x0).
  std::vector<RegId> Dsts;              ///< Registers written (no x0).
  MatrixInfo Mtx; ///< Matrix-engine details; Op is None unless MATRIX.

  Cycle FetchCycle = 0;    ///< Cycle it was fetched.
  Cycle DispatchCycle = 0; ///< Cycle it was dispatched.
  Cycle IssueCycle = 0;    ///< Cycle its execution started.
  /// First cycle a dependent may dispatch.
  Cycle ResultReadyCycle = Never;
  Cycle CompleteCycle = Never; ///< First cycle it may retire.
  Cycle RetireCycle = 0;       ///< Cycle it retired.

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
