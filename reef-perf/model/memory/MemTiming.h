//===- MemTiming.h - Memory timing models -----------------------*- C++ -*-===//
///
/// \file
/// Timing models of the memories, as plain classes so they can be tested
/// without a Sparta simulation. Memory (Memory.h) owns one per memory.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_MEMORY_MEMTIMING_H
#define REEF_PERF_MEMORY_MEMTIMING_H

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

/// The AXI master port and the memory behind it, as a black box: one beat per
/// cycle on the data channel, at most `MaxOutstanding` operations in flight,
/// and a fixed `Latency`. Reads and writes share one channel. With
/// `BlockingReads`, a read holds its requester until its read response
/// returns (occupancy = latency), as M3's blocking e-bus does; a write holds
/// it only while its beats go out.
class AxiModel {
public:
  /// Creates the port with a \p BytesPerBeat-wide data bus.
  AxiModel(uint32_t Latency, uint32_t BytesPerBeat, uint32_t MaxOutstanding,
           bool BlockingReads = false);

  /// Times \p Accesses, starting no earlier than \p Earliest, and books the
  /// data channel and an outstanding slot. \returns the timing.
  MemResponse access(std::span<const MemAccess> Accesses, Cycle Earliest);

  /// \returns true if a read holds its requester until its response.
  bool hasBlockingReads() const { return BlockingReads; }

  /// Keeps an outstanding slot booked \p Cycles longer from \p From, the end
  /// of an operation whose response comes later than access() timed it.
  void extend(Cycle From, uint64_t Cycles);

  /// \returns {axi_data, axi_outstanding}, for the end-of-run summary.
  std::vector<const ResourcePool *> getPools() const {
    return {&Data, &Outstanding};
  }

private:
  uint32_t Latency;
  uint32_t BytesPerBeat;
  bool BlockingReads;
  ResourcePool Data;        // One beat per cycle.
  ResourcePool Outstanding; // Each entry held until its operation is done.
};

/// One off-chip memory behind the SoC bus (XSPI flash, a PSRAM): one link
/// that each operation holds for a first-byte latency, then for its bytes at
/// `BytesPerCycle`. A read and a write can have different latencies. With a
/// burst limit, an operation longer than `MaxBurstBytes` is split into
/// bursts, and each burst pays the latency again. The core's e-bus accesses
/// and the DMA share the link. A rate of 0 means no link model: the M3
/// testbench's memory, which adds no delay (tests/systemc/Xbar.h).
class ExtMemModel {
public:
  /// Creates a link whose reads and writes both take \p Latency cycles to
  /// the first byte, with no burst limit. \throws std::invalid_argument if
  /// \p BytesPerCycle is negative.
  ExtMemModel(uint32_t Latency, double BytesPerCycle);

  /// Creates the link named \p Name in statistics. A read takes \p Latency
  /// cycles to its first byte and a write \p WriteLatency; a burst holds at
  /// most \p MaxBurstBytes (0: no limit). \throws std::invalid_argument if
  /// \p BytesPerCycle is negative.
  ExtMemModel(std::string Name, uint32_t Latency, uint32_t WriteLatency,
              double BytesPerCycle, uint32_t MaxBurstBytes);

  /// \returns true if the link is modeled (a nonzero rate).
  bool isTimed() const { return BytesPerCycle > 0; }

  /// One booking of the link.
  struct Slot {
    Cycle FirstByte = 0;       ///< When the first byte moves.
    Cycle Done = 0;            ///< The cycle after the last byte moves.
    uint32_t BurstLatency = 0; ///< First-byte cycles each of its bursts pays.
  };

  /// Books the link to read \p Bytes, or write them if \p IsWrite, starting
  /// no earlier than \p Earliest. \returns the booking; both cycles are
  /// \p Earliest if the link is not modeled.
  Slot transfer(uint64_t Bytes, Cycle Earliest, bool IsWrite = false);

  /// \returns the cycle after the first \p Bytes of \p Booked have moved.
  Cycle getReady(const Slot &Booked, uint64_t Bytes) const;

  /// \returns the link, for the end-of-run summary.
  const ResourcePool &getLink() const { return Link; }

private:
  // The cycles Bytes take at the link's rate, at least 1.
  uint64_t getDataCycles(uint64_t Bytes) const;
  // The bursts Bytes are split into, at least 1.
  uint64_t getBursts(uint64_t Bytes) const;

  uint32_t Latency;
  uint32_t WriteLatency;
  double BytesPerCycle;
  uint32_t MaxBurstBytes;
  ResourcePool Link;
};

} // namespace reef_perf

#endif // REEF_PERF_MEMORY_MEMTIMING_H
