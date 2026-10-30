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

#include <riscv/cfg.h>
#include <riscv/processor.h>
#include <riscv/simif.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <sstream>
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
  /// Not copyable, and so not movable: Spike's processor points into Config
  /// and Mem, which a copy or move would leave behind.
  FuncSim(const FuncSim &) = delete;
  FuncSim &operator=(const FuncSim &) = delete; ///< Not copyable.

  /// Executes and decodes the next instruction. `mpause` is reported but not
  /// executed, and ends the program.
  /// \returns the instruction, or nullptr once the program has halted.
  /// \throws std::runtime_error if the instruction traps.
  InstRef next();

  /// \returns true once next() has returned nullptr.
  bool isHalted() const { return Halted; }

  /// \returns the instructions handed out so far, including `mpause`.
  uint64_t getNumExecuted() const { return NumExecuted; }

  /// \returns the ELF's loaded segments, in file order.
  const std::vector<LoadSegment> &getLoadSegments() const { return Segments; }

private:
  /// Spike's view of the memory map: flat regions of host RAM. Everything
  /// else is unmapped (Spike raises an access fault); there is no MMIO. The
  /// overridden methods keep Spike's names.
  class FlatMemory final : public simif_t {
  public:
    FlatMemory(const std::vector<MemoryRegion> &Regions, const cfg_t *Config);
    char *addr_to_mem(reg_t PAddr) override;
    bool mmio_load(reg_t /*PAddr*/, size_t /*Len*/,
                   uint8_t * /*Bytes*/) override {
      return false;
    }
    bool mmio_store(reg_t /*PAddr*/, size_t /*Len*/,
                    const uint8_t * /*Bytes*/) override {
      return false;
    }
    void proc_reset(unsigned /*Id*/) override {}
    const cfg_t &get_cfg() const override { return *Config; }
    const std::map<size_t, processor_t *> &get_harts() const override {
      return Harts;
    }
    const char *get_symbol(uint64_t /*PAddr*/) override { return nullptr; }
    void setHart(processor_t *Proc) { Harts[0] = Proc; }

  private:
    struct Backing {
      MemoryRegion Region;
      std::vector<char> Bytes;
    };
    const cfg_t *Config; // Owned by FuncSim.
    std::vector<Backing> Backings;
    std::map<size_t, processor_t *> Harts;
  };

  /// Runs \p I in Spike and records its outcome, memory accesses and register
  /// writes.
  void execute(Inst &I);

  std::string ISA; // Config points into it.
  cfg_t Config;
  FlatMemory Mem;
  std::vector<LoadSegment> Segments;
  std::unique_ptr<std::FILE, int (*)(std::FILE *)> Log;
  std::ostringstream ConsoleOut;
  std::unique_ptr<processor_t> Proc;
  uint64_t NumExecuted = 0;
  bool Halted = false;
};

} // namespace reef_perf

#endif // REEF_PERF_FRONTEND_FUNCSIM_H
