//===- Tcm.h - A tightly coupled memory -------------------------*- C++ -*-===//
///
/// \file
/// A TCM (the ITCM or the DTCM): one single-port SRAM behind a fixed-priority
/// arbiter, as in M3's CoreAxi.scala. The port does one read or one write per
/// cycle. Priority, highest first: the core (fetch, the LSU), the AXI slave
/// (host, DMA), the debug module. The core never waits for a lower-priority
/// requester (`dbus.ready := true.B`).
///
/// Booking is synchronous, so requests arrive in simulation order, not cycle
/// order. A lower-priority transfer books the free cycles it can find; when a
/// higher-priority access later claims one of them, the transfer is bumped to
/// its next free cycle and its requester is told the new finish cycle through
/// MemRequest::OnFinishMoved.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_MEMORY_TCM_H
#define REEF_PERF_MEMORY_TCM_H

#include "common/Interfaces.h"
#include "common/ResourcePool.h"
#include "common/Types.h"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace reef_perf {

/// \returns the arbiter priority of \p Kind on a TCM port: 0 is highest.
uint32_t getTcmPriority(RequesterKind Kind);

/// Per-requester counters for one TCM.
struct TcmRequesterStats {
  uint64_t Reads = 0;        ///< Read operations.
  uint64_t Writes = 0;       ///< Write operations.
  uint64_t Bytes = 0;        ///< Bytes read or written.
  uint64_t PortCycles = 0;   ///< Cycles this requester held the port.
  uint64_t WaitCycles = 0;   ///< Cycles between Earliest and the last port
                             ///< cycle that this requester did not hold it.
  uint64_t BumpedCycles = 0; ///< Port cycles taken back by a higher priority.
};

/// One ITCM or DTCM: its port, its arbiter and its counters.
class Tcm {
public:
  /// Creates a TCM whose port is named \p Name in statistics. A line of
  /// \p LineBytes holds the port for \p CyclesPerLine cycles; the first line
  /// is usable \p Latency cycles after it starts.
  Tcm(std::string Name, uint32_t Latency, uint32_t LineBytes,
      uint32_t CyclesPerLine);

  /// Times \p Request, books the port and \returns the timing. A core access
  /// takes its cycles from any lower-priority transfer holding them.
  MemResponse access(const MemRequest &Request);

  /// A lower-priority transfer's timing and the handle delay() takes.
  struct Booking {
    MemResponse Response; ///< When the transfer runs.
    uint32_t Id;          ///< Its handle for delay().
  };

  /// Books \p Request, which must come from a lower-priority requester, and
  /// \returns its timing and handle.
  Booking book(const MemRequest &Request);

  /// Moves every port cycle of transfer \p Id before \p Earliest to the next
  /// cycles it may use, and reports a new finish through OnFinishMoved.
  void delay(uint32_t Id, Cycle Earliest);

  /// \returns the counters of \p Kind.
  const TcmRequesterStats &getStats(RequesterKind Kind) const;

  /// \returns the core's port bookings, for the end-of-run summary.
  const ResourcePool &getPort() const { return Port; }

private:
  /// A lower-priority transfer: the port cycles it holds.
  struct Transfer {
    RequesterKind Requester;
    std::set<Cycle> Cycles;
    std::function<void(Cycle)> OnFinishMoved;
  };

  MemResponse accessCore(const MemRequest &Request, uint64_t Occupancy);
  Booking accessLower(const MemRequest &Request, uint64_t Occupancy);
  MemResponse accessFetch(const MemRequest &Request, uint64_t Occupancy);
  // Marks Occupancy cycles from Start as core traffic, moving any
  // lower-priority transfer off them.
  void claimForCore(Cycle Start, uint64_t Occupancy);
  bool isCoreBusy(Cycle C) const;
  /// \returns whether a requester of \p Priority may take cycle \p C, which
  /// means no core booking and no lower-priority owner of equal or higher
  /// priority.
  bool isFreeFor(Cycle C, uint32_t Priority) const;
  /// Gives cycle \p C to transfer \p Id, bumping a lower-priority owner.
  void take(Cycle C, uint32_t Id);
  /// Takes cycle \p C back from its lower-priority owner and moves that
  /// transfer's port cycle to the next one it may use.
  void bump(Cycle C);
  /// Calls \p T's OnFinishMoved if its finish is no longer \p OldFinish.
  void reportMove(Transfer &T, Cycle OldFinish);
  Cycle getFinish(const Transfer &T) const;
  TcmRequesterStats &statsFor(RequesterKind Kind);

  uint32_t Latency;
  uint32_t LineBytes;
  uint32_t CyclesPerLine;
  ResourcePool Port;               // The core's bookings, in call order.
  std::map<Cycle, Cycle> CoreBusy; // Core bookings: start -> end (exclusive).
  std::map<Cycle, uint32_t> Owner; // Lower-priority port cycles -> transfer.
  std::vector<Transfer> Transfers;
  std::array<TcmRequesterStats, NumRequesterKinds> Stats{};
};

} // namespace reef_perf

#endif // REEF_PERF_MEMORY_TCM_H
