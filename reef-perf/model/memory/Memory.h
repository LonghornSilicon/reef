//===- Memory.h - The memory module -----------------------------*- C++ -*-===//
///
/// \file
/// The memory module: the ITCM, the DTCM and the AXI master port to off-core
/// memory (a black box). Other modules reach it through MemoryInterface. An
/// operation goes to the memory holding its first address, using the same
/// memory map as Spike; anything outside the ITCM and DTCM goes to AXI.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_MEMORY_MEMORY_H
#define REEF_PERF_MEMORY_MEMORY_H

#include "common/Interfaces.h"
#include "common/MemoryMap.h"
#include "common/ResourcePool.h"
#include "memory/MemTiming.h"

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

  /// \returns itcm, dtcm, axi_data and axi_outstanding.
  std::vector<const ResourcePool *> getPools() const;

private:
  std::vector<MemoryRegion> Regions = getDefaultMemoryMap();
  TcmModel ITCM;
  TcmModel DTCM;
  AxiModel AXI;
};

} // namespace reef_perf

#endif // REEF_PERF_MEMORY_MEMORY_H
