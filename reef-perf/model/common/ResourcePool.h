//===- ResourcePool.h - A pool of identical units ---------------*- C++ -*-===//
///
/// \file
/// A pool of identical execution resources, such as the four ALUs or the LSU
/// slot. An operation books one unit for its occupancy (1 = fully pipelined);
/// if every unit is busy, it starts when the earliest one frees up.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_COMMON_RESOURCEPOOL_H
#define REEF_PERF_COMMON_RESOURCEPOOL_H

#include <cstdint>
#include <string>
#include <vector>

namespace reef_perf {

/// A pool of identical units with per-unit busy-until times.
class ResourcePool {
public:
  /// Creates \p Count units named \p Name in statistics.
  /// \throws std::invalid_argument if \p Count is 0.
  ResourcePool(std::string Name, uint32_t Count);

  /// Books the earliest-free unit for \p Occupancy cycles, starting no
  /// earlier than \p Earliest. \returns the start cycle.
  uint64_t reserve(uint64_t Earliest, uint64_t Occupancy);

  /// \returns the cycle reserve() would start an operation that may begin at
  /// \p Earliest, without booking anything.
  uint64_t getNextFree(uint64_t Earliest) const;

  /// \returns the pool's name.
  const std::string &getName() const { return Name; }

  /// \returns the number of units.
  uint32_t getCount() const { return static_cast<uint32_t>(FreeAt.size()); }

  /// \returns the operations booked so far.
  uint64_t getOps() const { return Ops; }

  /// \returns the busy cycles booked so far, summed over all units.
  uint64_t getBusyCycles() const { return BusyCycles; }

private:
  std::string Name;
  std::vector<uint64_t> FreeAt; // Per unit: the first cycle it is free.
  uint64_t Ops = 0;
  uint64_t BusyCycles = 0;
};

} // namespace reef_perf

#endif // REEF_PERF_COMMON_RESOURCEPOOL_H
