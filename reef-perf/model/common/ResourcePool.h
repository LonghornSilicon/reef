//===- ResourcePool.h - A pool of identical resources -----------*- C++ -*-===//
///
/// \file
/// A pool of identical hardware resources, such as the four ALUs or the DTCM
/// port. An operation books one resource for its occupancy (1 = fully
/// pipelined); if every resource is busy, it starts when the earliest one frees
/// up. See docs/glossary.md for pool, resource and unit.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_COMMON_RESOURCEPOOL_H
#define REEF_PERF_COMMON_RESOURCEPOOL_H

#include "common/Types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace reef_perf {

/// A pool of identical resources with per-resource busy-until times.
class ResourcePool {
public:
  /// Creates \p Count resources named \p Name in statistics.
  /// \throws std::invalid_argument if \p Count is 0.
  ResourcePool(std::string Name, uint32_t Count);

  /// Books the earliest-free resource for \p Occupancy cycles, starting no
  /// earlier than \p Earliest. \returns the start cycle.
  Cycle reserve(Cycle Earliest, uint64_t Occupancy);

  /// Like reserve(), but books the first resource, for an operation only it
  /// can run. \returns the start cycle.
  Cycle reserveFirst(Cycle Earliest, uint64_t Occupancy);

  /// \returns the cycle reserve() would start an operation that may begin at
  /// \p Earliest, without booking anything.
  Cycle getNextFree(Cycle Earliest) const;

  /// \returns the pool's name.
  const std::string &getName() const { return Name; }

  /// \returns the number of resources.
  uint32_t getCount() const { return static_cast<uint32_t>(FreeAt.size()); }

  /// \returns the operations booked so far.
  uint64_t getOps() const { return Ops; }

  /// \returns the busy cycles booked so far, summed over all resources.
  uint64_t getBusyCycles() const { return BusyCycles; }

private:
  Cycle book(Cycle &FreeAtCycle, Cycle Earliest, uint64_t Occupancy);

  std::string Name;
  std::vector<Cycle> FreeAt; // Per resource: the first cycle it is free.
  uint64_t Ops = 0;
  uint64_t BusyCycles = 0;
};

} // namespace reef_perf

#endif // REEF_PERF_COMMON_RESOURCEPOOL_H
