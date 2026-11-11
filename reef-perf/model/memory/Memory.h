//===- Memory.h - The memory module -----------------------------*- C++ -*-===//
///
/// \file
/// The memory module: the ITCM, the DTCM (memory/Tcm.h), the AXI master
/// port (the e-bus) and the off-chip memory behind it (ExtMemModel). Other
/// modules reach it through MemoryInterface. An operation goes to the memory
/// holding its first address, using the same memory map as Spike; anything
/// outside the ITCM and DTCM goes to AXI.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_MEMORY_MEMORY_H
#define REEF_PERF_MEMORY_MEMORY_H

#include "common/Interfaces.h"
#include "common/MemoryMap.h"
#include "common/ResourcePool.h"
#include "memory/Dma.h"
#include "memory/Host.h"
#include "memory/MemTiming.h"
#include "memory/Tcm.h"

#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace reef_perf {

/// The memory module. Tree location: top.memory.
class Memory : public sparta::Unit, public MemoryInterface {
public:
  // Parameter names are config keys, so they keep Sparta's snake_case.
  // NOLINTBEGIN(readability-identifier-naming)
  struct MemoryParameterSet : public sparta::ParameterSet {
    explicit MemoryParameterSet(sparta::TreeNode *Node)
        : sparta::ParameterSet(Node) {}
    PARAMETER(uint32_t, itcm_latency, 2, "ITCM latency of a single-line access")
    PARAMETER(uint32_t, itcm_line_bytes, 16, "ITCM bytes per transfer")
    PARAMETER(uint32_t, itcm_cycles_per_line, 1,
              "Cycles the ITCM port is busy per line")
    PARAMETER(uint32_t, dtcm_latency, 2, "DTCM latency of a single-line access")
    PARAMETER(uint32_t, dtcm_line_bytes, 16, "DTCM bytes per transfer")
    PARAMETER(uint32_t, dtcm_cycles_per_line, 1,
              "Cycles the DTCM port is busy per line")
    PARAMETER(uint32_t, axi_latency, 20,
              "Round-trip latency of a one-beat AXI operation")
    PARAMETER(uint32_t, axi_bytes_per_beat, 32, "AXI data bus width in bytes")
    PARAMETER(uint32_t, axi_max_outstanding, 4,
              "AXI operations in flight at once")
    PARAMETER(bool, axi_blocking_reads, false,
              "A read holds its requester until its AXI read response returns")
    PARAMETER(uint32_t, dma_setup_cycles, 20,
              "Cycles from a DMA GO store to its first beat")
    PARAMETER(uint32_t, dma_beat_bytes, 16, "Bytes per DMA beat into a TCM")
    PARAMETER(uint32_t, dma_queue_depth, 0,
              "Unfinished DMA transfers before a GO store waits (0: no limit)")
    PARAMETER(bool, host_boot, false,
              "Time the host's ITCM load before the core leaves reset")
    PARAMETER(bool, host_boot_data, false,
              "With host_boot, also time the host's DTCM data load")
    PARAMETER(uint32_t, host_setup_cycles, 100,
              "Cycles before the host's first boot byte")
    PARAMETER(uint32_t, host_cycles_per_byte, 8,
              "Cycles per byte on the host link")
    PARAMETER(uint32_t, ext_mem_latency, 0,
              "External memory link cycles to the first byte of an operation")
    PARAMETER(double, ext_mem_bytes_per_cycle, 0,
              "External memory link bytes per core cycle (0: not modeled)")
    PARAMETER(uint32_t, ext_mem_max_burst_bytes, 0,
              "External memory link bytes per burst; each burst pays the "
              "latency (0: no limit)")
    PARAMETER(uint32_t, ext_ram_mb, 0,
              "Size in MB of the second external memory, a RAM with its own "
              "link (0: none)")
    PARAMETER(uint32_t, ext_ram_base, 0x24000000,
              "First address of the second external memory")
    PARAMETER(uint32_t, ext_ram_latency, 0,
              "RAM link cycles to the first byte of a read burst")
    PARAMETER(uint32_t, ext_ram_write_latency, 0,
              "RAM link cycles to the first byte of a write burst")
    PARAMETER(double, ext_ram_bytes_per_cycle, 0,
              "RAM link bytes per core cycle (0: not modeled)")
    PARAMETER(uint32_t, ext_ram_max_burst_bytes, 0,
              "RAM link bytes per burst; each burst pays the latency (0: no "
              "limit)")
  };

  /// Sparta resource name; ResourceFactory requires this exact member name.
  static constexpr const char *name = "memory";
  // NOLINTEND(readability-identifier-naming)

  /// Creates the unit on \p Node with parameters \p Params.
  Memory(sparta::TreeNode *Node, const MemoryParameterSet *Params);

  /// Sets the memory map to \p NewRegions, the regions Spike was given.
  void setMemoryMap(std::vector<MemoryRegion> NewRegions) {
    Regions = std::move(NewRegions);
  }

  /// \returns the kind of region holding \p Addr, or RegionKind::External.
  RegionKind getRegionKind(Address Addr) const;

  /// Times \p Request and books the memory it uses. \returns its timing.
  MemResponse access(const MemRequest &Request) override;

  /// \returns itcm, itcm_fetch, dtcm, axi_data, axi_outstanding and, when
  /// modeled, ext_mem_link and ext_ram_link.
  std::vector<const ResourcePool *> getPools() const;

  /// \returns the DMA, for its statistics.
  const Dma &getDma() const { return DMA; }

  /// Sets the host's boot load from the ELF's \p Segments: the bytes that
  /// fall in an ITCM (code) and in a DTCM (data).
  void setBootImage(const std::vector<LoadSegment> &Segments);

  /// \returns the host, for its boot cycles.
  const Host &getHost() const { return TheHost; }

private:
  /// Adds the external memory link's time to \p Response, the e-bus timing
  /// of \p Request. \returns the result.
  MemResponse addExtMemTime(const MemRequest &Request, MemResponse Response);
  // The link Addr is behind: the RAM's for an address in the second external
  // memory, otherwise the first memory's.
  ExtMemModel *getLink(Address Addr);

  std::vector<MemoryRegion> Regions = getDefaultMemoryMap();
  Tcm ITCM;
  Tcm DTCM;
  AxiModel AXI;
  ExtMemModel Ext;
  ExtMemModel Ram;
  Address RamBase;
  uint64_t RamBytes;
  Dma DMA;
  Host TheHost;
};

} // namespace reef_perf

#endif // REEF_PERF_MEMORY_MEMORY_H
