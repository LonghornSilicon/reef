//===- MemoryMap.h - Reef's physical memory map -----------------*- C++ -*-===//
///
/// \file
/// Reef's physical memory map. Spike backs each region with host RAM and the
/// memory module uses the same list to pick which memory times an access, so
/// the two can never disagree about where an address lives.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_COMMON_MEMORYMAP_H
#define REEF_PERF_COMMON_MEMORYMAP_H

#include "common/Types.h"

#include <cstdint>
#include <vector>

namespace reef_perf {

/// What a region of the memory map is.
enum class RegionKind : uint8_t {
  ITCM,    ///< Instruction tightly-coupled memory.
  DTCM,    ///< Data tightly-coupled memory.
  External ///< Off-core memory, reached over the AXI master port.
};

/// A contiguous range of physical memory.
struct MemoryRegion {
  Address Start = 0;                      ///< First byte address.
  uint32_t Length = 0;                    ///< Size in bytes.
  RegionKind Kind = RegionKind::External; ///< What it is; Spike ignores it.

  /// \returns true if Start <= \p Addr < Start + Length.
  bool contains(Address Addr) const {
    return Addr >= Start && Addr - Start < Length;
  }
};

/// \returns M3's default map: 8 KB ITCM at 0x0, 32 KB DTCM at 0x10000.
inline std::vector<MemoryRegion> getDefaultMemoryMap() {
  return {{0x00000000, 0x2000, RegionKind::ITCM},
          {0x00010000, 0x8000, RegionKind::DTCM}};
}

/// \returns M3's highmem map: 1 MB ITCM at 0x0, 1 MB DTCM at 0x100000.
inline std::vector<MemoryRegion> getHighmemMemoryMap() {
  constexpr uint32_t OneMiB = 1024 * 1024;
  return {{0x00000000, OneMiB, RegionKind::ITCM},
          {0x00100000, OneMiB, RegionKind::DTCM}};
}

} // namespace reef_perf

#endif // REEF_PERF_COMMON_MEMORYMAP_H
