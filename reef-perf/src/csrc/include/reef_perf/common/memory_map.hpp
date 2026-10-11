#pragma once

/** @file
 *  @brief Reef's physical memory map, shared by Spike and the memory module.
 *
 *  Spike backs each region with host RAM; the memory module uses the same
 *  regions to decide which memory times an access. Keeping one list means
 *  the two can never disagree about where an address lives.
 */

#include <cstdint>
#include <vector>

namespace reef_perf {

/// What a region of the memory map is.
enum class RegionKind : std::uint8_t {
    ITCM, ///< Instruction tightly-coupled memory.
    DTCM, ///< Data tightly-coupled memory.
    EXT,  ///< Off-core memory, reached over the AXI master port.
};

/// A contiguous range of physical memory.
struct MemoryRegion {
    /// First byte address of the region.
    std::uint32_t start = 0;
    /// Size of the region in bytes.
    std::uint32_t length = 0;
    /// What the region is. Spike ignores it.
    RegionKind kind = RegionKind::EXT;

    /** Whether an address falls inside the region.
     *
     *  @param addr Byte address.
     *  @return True if start <= addr < start + length.
     */
    [[nodiscard]] bool contains(std::uint32_t addr) const {
        return addr >= start && addr - start < length;
    }
};

/** Reef's M3 default map: 8 KB ITCM at 0x0, 32 KB DTCM at 0x10000.
 *
 *  @return The regions.
 */
inline std::vector<MemoryRegion> default_memory_map() {
    return {{0x00000000, 0x2000, RegionKind::ITCM},
            {0x00010000, 0x8000, RegionKind::DTCM}};
}

/** The M3 highmem map: 1 MB ITCM at 0x0, 1 MB DTCM at 0x100000.
 *
 *  @return The regions.
 */
inline std::vector<MemoryRegion> highmem_memory_map() {
    constexpr std::uint32_t kOneMiB = 1024 * 1024;
    return {{0x00000000, kOneMiB, RegionKind::ITCM},
            {0x00100000, kOneMiB, RegionKind::DTCM}};
}

} // namespace reef_perf
