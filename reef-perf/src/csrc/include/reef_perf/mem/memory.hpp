#pragma once

/** @file
 *  @brief Memory module: the TCMs and the AXI port to off-core memory.
 *         Tree location: top.mem.
 *
 *  Units:
 *  - top.mem.tcm  Tcm (ITCM and DTCM)
 *  - top.mem.axi  Axi (AXI master port and a black box behind it)
 *
 *  Other modules reach memory through MemoryInterface. An operation is sent
 *  to the memory that holds its first address, using the same memory map as
 *  Spike: ITCM and DTCM regions go to the TCMs, everything else to AXI.
 */

#include "reef_perf/common/interfaces.hpp"
#include "reef_perf/common/memory_map.hpp"
#include "reef_perf/common/module.hpp"
#include "reef_perf/common/resource_pool.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace reef_perf {

class Axi;
class Tcm;

/// The memory module.
class Memory : public Module, public MemoryInterface {
  public:
    /// Creates the module; its tree node is top.mem.
    Memory();

    /** Registers the factories of the memory units.
     *
     *  @param resources The simulation's resource set.
     */
    void add_factories(sparta::ResourceSet& resources) override;

    /// Looks up the memory units.
    void bind() override;

    /** The module's pools.
     *
     *  @return itcm, dtcm, axi_data, axi_outstanding.
     */
    [[nodiscard]] std::vector<const ResourcePool*> pools() const override;

    /** Sets the memory map. Call before the simulation runs.
     *
     *  @param regions The regions Spike was given.
     */
    void set_memory_map(std::vector<MemoryRegion> regions) {
        regions_ = std::move(regions);
    }

    /** Which memory holds an address.
     *
     *  @param addr Byte address.
     *  @return The kind of the region containing it, or RegionKind::EXT.
     */
    [[nodiscard]] RegionKind region_of(std::uint32_t addr) const;

    /** Times one memory operation and books the resources it uses.
     *
     *  @param request The operation.
     *  @return Its timing.
     */
    MemResponse access(const MemRequest& request) override;

  protected:
    /** The module's units.
     *
     *  @return {"tcm", "axi"}.
     */
    [[nodiscard]] std::vector<std::string> unit_names() const override;

  private:
    /// The memory map.
    std::vector<MemoryRegion> regions_ = default_memory_map();
    /// The TCMs, valid after bind().
    Tcm* tcm_ = nullptr;
    /// The AXI port, valid after bind().
    Axi* axi_ = nullptr;
};

} // namespace reef_perf
