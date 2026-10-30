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

/// \returns the DMA's register block (memory/Dma.h): external, so its
/// stores and loads go out through the e-bus. Not in M3; the address is an
/// assumption, outside the M3 testbench's memory (0x20000000, 4 MB).
inline MemoryRegion getDmaRegsRegion() {
  return {0x40000000, 0x20, RegionKind::External};
}

/// \returns M3's default map: 8 KB ITCM at 0x0, 32 KB DTCM at 0x10000, plus
/// the DMA's register block.
inline std::vector<MemoryRegion> getDefaultMemoryMap() {
  return {{0x00000000, 0x2000, RegionKind::ITCM},
          {0x00010000, 0x8000, RegionKind::DTCM},
          getDmaRegsRegion()};
}

/// \returns M3's highmem map: 1 MB ITCM at 0x0, 1 MB DTCM at 0x100000.
inline std::vector<MemoryRegion> getHighmemMemoryMap() {
  constexpr uint32_t OneMiB = 1024 * 1024;
  return {{0x00000000, OneMiB, RegionKind::ITCM},
          {0x00100000, OneMiB, RegionKind::DTCM},
          getDmaRegsRegion()};
}

} // namespace reef_perf

#endif // REEF_PERF_COMMON_MEMORYMAP_H
