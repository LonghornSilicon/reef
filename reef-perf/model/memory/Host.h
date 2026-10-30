//===- Host.h - The external host -------------------------------*- C++ -*-===//
///
/// \file
/// The external host's boot load. M3 holds the core in reset with its clock
/// gated until the host releases it (CoreAxiCSR.scala, `resetReg` starts at
/// 3). Before that, the host writes the program's ITCM image through the AXI
/// slave and, with `host_boot_data`, its data: the DTCM image, which holds
/// the firmware's mailbox and so the prompt. Then it releases the core with
/// one 4-byte write to the reset register. The core makes no requests while
/// in reset, so the host gets every TCM cycle and the arbiter does not
/// matter. Data for other memories (HyperRAM) is not modeled yet.
///
/// Boot cycles = `host_setup_cycles` + (bytes + 4) x `host_cycles_per_byte`,
/// one link for code and data. Both are assumptions: there is no Reef source
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
  /// starts, and the data as well if \p LoadData: \p SetupCycles once, then
  /// \p CyclesPerByte per byte.
  Host(bool Enabled, bool LoadData, uint32_t SetupCycles,
       uint32_t CyclesPerByte);

  /// Sets the bytes the boot image holds: \p ItcmBytes of code and
  /// \p DataBytes of data (the DTCM image).
  void setBootImage(uint64_t ItcmBytes, uint64_t DataBytes);

  /// \returns the cycles from power-up until the core leaves reset, or 0 if
  /// the boot load is off.
  Cycle getBootCycles() const { return BootCycles; }

  /// \returns the bytes the boot load wrote, the reset write included.
  uint64_t getBootBytes() const { return BootBytes; }

  /// \returns the data bytes the boot load wrote.
  uint64_t getBootDataBytes() const { return BootDataBytes; }

private:
  bool Enabled;
  bool LoadData;
  uint32_t SetupCycles;
  uint32_t CyclesPerByte;
  Cycle BootCycles = 0;
  uint64_t BootBytes = 0;
  uint64_t BootDataBytes = 0;
};

} // namespace reef_perf

#endif // REEF_PERF_MEMORY_HOST_H
