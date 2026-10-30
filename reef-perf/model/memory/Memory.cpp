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
          Params->axi_max_outstanding),
      DMA(Params->dma_setup_cycles, Params->dma_beat_bytes,
          [this](Address Addr) -> Tcm * {
            switch (getRegionKind(Addr)) {
            case RegionKind::ITCM:
              return &ITCM;
            case RegionKind::DTCM:
              return &DTCM;
            case RegionKind::External:
              break;
            }
            return nullptr;
          }),
      TheHost(Params->host_boot, Params->host_setup_cycles,
              Params->host_cycles_per_byte) {}

void Memory::setBootImage(const std::vector<LoadSegment> &Segments) {
  uint64_t ItcmBytes = 0;
  for (const LoadSegment &Segment : Segments)
    if (getRegionKind(Segment.Addr) == RegionKind::ITCM)
      ItcmBytes += Segment.FileBytes;
  TheHost.setBootImage(ItcmBytes);
}

RegionKind Memory::getRegionKind(Address Addr) const {
  for (const MemoryRegion &Region : Regions)
    if (Region.contains(Addr))
      return Region.Kind;
  return RegionKind::External;
}

MemResponse Memory::access(const MemRequest &Request) {
  // Fetch books only the ITCM; fetch over AXI (outside the ITCM) is not
  // modeled, as before.
  if (Request.Requester == RequesterKind::IFetch &&
      (Request.Accesses.empty() ||
       getRegionKind(Request.Accesses.front().Addr) != RegionKind::ITCM))
    return {Request.Earliest, 1, 1};
  // An empty access list (e.g. a vector op with vl = 0) is one DTCM line.
  RegionKind Kind = Request.Accesses.empty()
                        ? RegionKind::DTCM
                        : getRegionKind(Request.Accesses.front().Addr);
  switch (Kind) {
  case RegionKind::ITCM:
    return ITCM.access(Request);
  case RegionKind::DTCM:
    return DTCM.access(Request);
  case RegionKind::External:
    break;
  }
  MemResponse Response = AXI.access(Request.Accesses, Request.Earliest);
  const MemoryRegion DmaRegs = getDmaRegsRegion();
  if (Request.Accesses.empty() ||
      !DmaRegs.contains(Request.Accesses.front().Addr))
    return Response;
  // A register access reaches the DMA when its AXI operation is done.
  const Cycle Arrive = Response.Start + Response.Latency;
  for (const MemAccess &Access : Request.Accesses) {
    const uint32_t Offset = Access.Addr - DmaRegs.Start;
    if (Access.IsStore)
      DMA.writeRegister(Offset, Access.Value, Arrive);
    else if (Offset == DmaWait && DMA.getFinish() > Arrive)
      Response.Latency = DMA.getFinish() - Response.Start;
  }
  return Response;
}

std::vector<const ResourcePool *> Memory::getPools() const {
  std::vector<const ResourcePool *> Pools = {&ITCM.getPort(), &DTCM.getPort()};
  for (const ResourcePool *Pool : AXI.getPools())
    Pools.push_back(Pool);
  return Pools;
}
