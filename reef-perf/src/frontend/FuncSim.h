//===- FuncSim.h - Functional simulator (Spike) -----------------*- C++ -*-===//
///
/// \file
/// The functional simulator: runs the program in Spike one instruction at a
/// time and returns each instruction already executed and decoded
/// (execute-at-fetch). The timing model only decides *when* things happen.
/// Reef is in-order and never executes wrong-path instructions, so Spike never
/// has to be rolled back.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_FRONTEND_FUNCSIM_H
#define REEF_PERF_FRONTEND_FUNCSIM_H

#include "common/Inst.h"
#include "common/MemoryMap.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace reef_perf {

/// Encoding of `mpause`, the instruction that ends a program on Reef.
constexpr uint32_t MpauseEncoding = 0x08000073;

/// ISA string for Reef's M3 baseline: RV32IMF + Zbb + Zve32f, VLEN = 128.
constexpr const char *DefaultISA = "rv32imf_zicsr_zifencei_zbb_zve32f_zvl128b";

/// Configuration of the simulated machine.
struct SpikeOptions {
  std::string ISA = DefaultISA; ///< ISA string passed to Spike.
  /// Memory backed by host RAM; accesses elsewhere raise access faults.
  std::vector<MemoryRegion> Regions = getDefaultMemoryMap();
};

/// Runs a RISC-V ELF in Spike and hands out decoded instructions in order.
class FuncSim {
public:
  /// Loads the 32-bit RISC-V ELF \p ElfPath into a machine configured by
  /// \p Options. \throws std::runtime_error if the ELF cannot be loaded.
  FuncSim(const std::string &ElfPath, const SpikeOptions &Options);
  ~FuncSim();                                   ///< Releases Spike.
  FuncSim(const FuncSim &) = delete;            ///< Not copyable.
  FuncSim &operator=(const FuncSim &) = delete; ///< Not copyable.
  FuncSim(FuncSim &&) noexcept;                 ///< Movable.
  FuncSim &operator=(FuncSim &&) noexcept;      ///< Movable.

  /// Executes and decodes the next instruction. `mpause` is reported but not
  /// executed, and ends the program.
  /// \returns the instruction, or nullptr once the program has halted.
  /// \throws std::runtime_error if the instruction traps.
  InstRef next();

  /// \returns true once next() has returned nullptr.
  bool isHalted() const;

  /// \returns the instructions handed out so far, including `mpause`.
  uint64_t getNumExecuted() const;

private:
  class Impl; // Spike state, kept out of this header.
  std::unique_ptr<Impl> TheImpl;
};

} // namespace reef_perf

#endif // REEF_PERF_FRONTEND_FUNCSIM_H
