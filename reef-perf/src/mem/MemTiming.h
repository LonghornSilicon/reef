//===- MemTiming.h - Memory timing models -----------------------*- C++ -*-===//
///
/// \file
/// Timing models of the memories, as plain classes so they can be tested
/// without a Sparta simulation. Memory (Memory.h) owns one per memory.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_MEM_MEMTIMING_H
#define REEF_PERF_MEM_MEMTIMING_H

#include "common/Inst.h"
#include "common/Interfaces.h"
#include "common/ResourcePool.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace reef_perf {

/// \returns the distinct \p BlockBytes-aligned blocks \p Accesses touch, at
/// least 1 (an empty list counts as one block). An access may straddle two
/// blocks. \throws std::invalid_argument if \p BlockBytes is 0.
uint32_t countDistinctBlocks(std::span<const MemAccess> Accesses,
                             uint32_t BlockBytes);

/// An ITCM or DTCM: one port moving a line every `CyclesPerLine` cycles, with
/// latency = `Latency` + transfer cycles - 1. No banks.
class TcmModel {
public:
  /// Creates a TCM whose port is named \p Name in statistics.
  TcmModel(std::string Name, uint32_t Latency, uint32_t LineBytes,
           uint32_t CyclesPerLine);

  /// Times \p Accesses, starting no earlier than \p Earliest, and books the
  /// port. \returns the timing.
  MemResponse access(std::span<const MemAccess> Accesses, uint64_t Earliest);

  /// \returns the port, for the end-of-run summary.
  const ResourcePool &getPort() const { return Port; }

private:
  uint32_t Latency;
  uint32_t LineBytes;
  uint32_t CyclesPerLine;
  ResourcePool Port;
};

/// The AXI master port and the memory behind it, as a black box: one beat per
/// cycle on the data channel, at most `MaxOutstanding` operations in flight,
/// and a fixed `Latency`. Reads and writes share one channel.
class AxiModel {
public:
  /// Creates the port with a \p BytesPerBeat-wide data bus.
  AxiModel(uint32_t Latency, uint32_t BytesPerBeat, uint32_t MaxOutstanding);

  /// Times \p Accesses, starting no earlier than \p Earliest, and books the
  /// data channel and an outstanding slot. \returns the timing.
  MemResponse access(std::span<const MemAccess> Accesses, uint64_t Earliest);

  /// \returns {axi_data, axi_outstanding}, for the end-of-run summary.
  std::vector<const ResourcePool *> getPools() const {
    return {&Data, &Outstanding};
  }

private:
  uint32_t Latency;
  uint32_t BytesPerBeat;
  ResourcePool Data;        // One beat per cycle.
  ResourcePool Outstanding; // Each slot held until its operation is done.
};

} // namespace reef_perf

#endif // REEF_PERF_MEM_MEMTIMING_H
