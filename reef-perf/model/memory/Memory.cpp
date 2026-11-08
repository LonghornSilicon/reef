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
          Params->axi_max_outstanding, Params->axi_blocking_reads),
      Ext(Params->ext_mem_latency, Params->ext_mem_bytes_per_cycle),
      DMA(
          Params->dma_setup_cycles, Params->dma_beat_bytes, &Ext,
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
          },
          Params->dma_queue_depth) {}

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
    return addExtMemTime(Request, Response);
  // A register access reaches the DMA when its AXI operation is done.
  const Cycle Arrive = Response.Start + Response.Latency;
  for (const MemAccess &Access : Request.Accesses) {
    const uint32_t Offset = Access.Addr - DmaRegs.Start;
    if (Access.IsStore) {
      // A GO into a full queue holds the e-bus until the DMA takes it.
      const Cycle Accept = DMA.writeRegister(Offset, Access.Value, Arrive);
      if (Accept > Arrive) {
        Response.Latency = Accept - Response.Start;
        Response.Occupancy = Response.Latency;
      }
    } else if (Offset == DmaWait && DMA.getFinish() > Arrive)
      Response.Latency = DMA.getFinish() - Response.Start;
  }
  return Response;
}

MemResponse Memory::addExtMemTime(const MemRequest &Request,
                                  MemResponse Response) {
  if (!Ext.isTimed())
    return Response;
  uint64_t Bytes = 0;
  for (const MemAccess &Access : Request.Accesses)
    Bytes += Access.Size;
  // The link time adds to the e-bus path's own latency (axi_latency). A
  // blocking read keeps holding its requester until the data is back.
  const bool Blocking =
      AXI.hasBlockingReads() &&
      (Request.Accesses.empty() || !Request.Accesses.front().IsStore);
  const Cycle LinkDone = Ext.transfer(Bytes, Response.Start);
  Response.Latency += LinkDone - Response.Start;
  if (Blocking)
    Response.Occupancy = Response.Latency;
  return Response;
}

std::vector<const ResourcePool *> Memory::getPools() const {
  std::vector<const ResourcePool *> Pools = {&ITCM.getPort(), &DTCM.getPort()};
  for (const ResourcePool *Pool : AXI.getPools())
    Pools.push_back(Pool);
  if (Ext.isTimed())
    Pools.push_back(&Ext.getLink());
  return Pools;
}
