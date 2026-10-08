#include "reef_perf/mem/memory.hpp"

#include "reef_perf/mem/axi.hpp"
#include "reef_perf/mem/tcm.hpp"

#include "sparta/simulation/ResourceFactory.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace reef_perf {

Memory::Memory() : Module("mem", "TCMs and the AXI port") {}

void Memory::add_factories(sparta::ResourceSet& resources) {
    resources.addResourceFactory<
        sparta::ResourceFactory<Tcm, Tcm::TcmParameterSet>>();
    resources.addResourceFactory<
        sparta::ResourceFactory<Axi, Axi::AxiParameterSet>>();
}

std::vector<std::string> Memory::unit_names() const {
    return {Tcm::name, Axi::name};
}

void Memory::bind() {
    tcm_ = unit<Tcm>(Tcm::name);
    axi_ = unit<Axi>(Axi::name);
}

std::vector<const ResourcePool*> Memory::pools() const {
    std::vector<const ResourcePool*> all = tcm_->pools();
    for (const ResourcePool* pool : axi_->pools()) {
        all.push_back(pool);
    }
    return all;
}

RegionKind Memory::region_of(std::uint32_t addr) const {
    for (const MemoryRegion& region : regions_) {
        if (region.contains(addr)) {
            return region.kind;
        }
    }
    return RegionKind::EXT;
}

MemResponse Memory::access(const MemRequest& request) {
    const RegionKind kind = request.accesses.empty()
                                ? RegionKind::DTCM
                                : region_of(request.accesses.front().addr);
    if (kind == RegionKind::EXT) {
        return axi_->access(request.accesses, request.earliest);
    }
    return tcm_->access(kind, request.accesses, request.earliest);
}

} // namespace reef_perf
