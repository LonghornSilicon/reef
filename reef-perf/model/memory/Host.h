//===- Host.h - The external host -------------------------------*- C++ -*-===//
///
/// \file
/// The external host's boot load. M3 holds the core in reset with its clock
/// gated until the host releases it (CoreAxiCSR.scala, `resetReg` starts at
/// 3). Before that, the host writes the program's ITCM image through the AXI
/// slave, then releases the core with one 4-byte write to the reset
/// register. The core makes no requests while in reset, so the host gets
/// every ITCM cycle and the arbiter does not matter.
///
/// Boot cycles = `host_setup_cycles` + (ITCM bytes + 4) x
/// `host_cycles_per_byte`. Both are assumptions: there is no Reef source
/// for the host link's speed. Off by default (`host_boot`), so the core
/// starts in cycle 0 as before.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_MEMORY_HOST_H
#define REEF_PERF_MEMORY_HOST_H

#include "common/Types.h"

#include <cstdint>

namespace reef_perf {

/// The host. Owned by the memory module.
class Host {
public:
  /// Creates a host that, if \p Enabled, loads the ITCM before the core
  /// starts: \p SetupCycles once, then \p CyclesPerByte per byte.
  Host(bool Enabled, uint32_t SetupCycles, uint32_t CyclesPerByte);

  /// Sets the bytes the host writes into the ITCM at boot to \p ItcmBytes.
  void setBootImage(uint64_t ItcmBytes);

  /// \returns the cycles from power-up until the core leaves reset, or 0 if
  /// the boot load is off.
  Cycle getBootCycles() const { return BootCycles; }

  /// \returns the bytes the boot load wrote, the reset write included.
  uint64_t getBootBytes() const { return BootBytes; }

private:
  bool Enabled;
  uint32_t SetupCycles;
  uint32_t CyclesPerByte;
  Cycle BootCycles = 0;
  uint64_t BootBytes = 0;
};

} // namespace reef_perf

#endif // REEF_PERF_MEMORY_HOST_H
