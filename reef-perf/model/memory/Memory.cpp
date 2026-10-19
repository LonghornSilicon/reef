//===- Memory.cpp - The memory module -------------------------------------===//

#include "memory/Memory.h"

#include <cstdint>
#include <vector>

using namespace reef_perf;

Memory::Memory(sparta::TreeNode *Node, const MemoryParameterSet *Params)
    : sparta::Unit(Node, name),
      ITCM("itcm", Params->itcm_latency, Params->itcm_line_bytes,
           Params->itcm_cycles_per_line),
      DTCM("dtcm", Params->dtcm_latency, Params->dtcm_line_bytes,
           Params->dtcm_cycles_per_line),
      AXI(Params->axi_latency, Params->axi_bytes_per_beat,
          Params->axi_max_outstanding) {}

RegionKind Memory::getRegionKind(Address Addr) const {
  for (const MemoryRegion &Region : Regions)
    if (Region.contains(Addr))
      return Region.Kind;
  return RegionKind::External;
}

MemResponse Memory::access(const MemRequest &Request) {
  // An empty access list (e.g. a vector op with vl = 0) is one DTCM line.
  RegionKind Kind = Request.Accesses.empty()
                        ? RegionKind::DTCM
                        : getRegionKind(Request.Accesses.front().Addr);
  switch (Kind) {
  case RegionKind::ITCM:
    return ITCM.access(Request.Accesses, Request.Earliest);
  case RegionKind::DTCM:
    return DTCM.access(Request.Accesses, Request.Earliest);
  case RegionKind::External:
    break;
  }
  return AXI.access(Request.Accesses, Request.Earliest);
}

std::vector<const ResourcePool *> Memory::getPools() const {
  std::vector<const ResourcePool *> Pools = {&ITCM.getPort(), &DTCM.getPort()};
  for (const ResourcePool *Pool : AXI.getPools())
    Pools.push_back(Pool);
  return Pools;
}
