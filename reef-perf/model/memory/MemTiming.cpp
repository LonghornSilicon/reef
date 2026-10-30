//===- MemTiming.cpp - Memory timing models -------------------------------===//

#include "memory/MemTiming.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

using namespace reef_perf;

uint32_t reef_perf::countDistinctBlocks(std::span<const MemAccess> Accesses,
                                        uint32_t BlockBytes) {
  if (BlockBytes == 0)
    throw std::invalid_argument("block size must be at least 1 byte");
  if (Accesses.empty())
    return 1;
  std::unordered_set<uint32_t> Blocks;
  for (const MemAccess &Access : Accesses) {
    Blocks.insert(Access.Addr / BlockBytes);
    Blocks.insert((Access.Addr + Access.Size - 1) / BlockBytes);
  }
  return static_cast<uint32_t>(Blocks.size());
}

AxiModel::AxiModel(uint32_t Latency, uint32_t BytesPerBeat,
                   uint32_t MaxOutstanding, bool BlockingReads)
    : Latency(Latency), BytesPerBeat(BytesPerBeat),
      BlockingReads(BlockingReads), Data("axi_data", 1),
      Outstanding("axi_outstanding", MaxOutstanding) {}

MemResponse AxiModel::access(std::span<const MemAccess> Accesses,
                             Cycle Earliest) {
  MemResponse Response;
  Response.Occupancy = countDistinctBlocks(Accesses, BytesPerBeat);
  Response.Latency = Latency + Response.Occupancy - 1;
  // Start when both the data channel and an outstanding slot are free.
  Response.Start =
      std::max(Data.getNextFree(Earliest), Outstanding.getNextFree(Earliest));
  Data.reserve(Response.Start, Response.Occupancy);
  Outstanding.reserve(Response.Start, Response.Latency);
  // A blocking read keeps its requester until the read response is back.
  const bool IsRead = Accesses.empty() || !Accesses.front().IsStore;
  if (BlockingReads && IsRead)
    Response.Occupancy = Response.Latency;
  return Response;
}
