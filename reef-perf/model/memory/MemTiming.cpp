//===- MemTiming.cpp - Memory timing models -------------------------------===//

#include "memory/MemTiming.h"

#include <algorithm>
#include <cmath>
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

void AxiModel::extend(Cycle From, uint64_t Cycles) {
  // Bookings are made in call order, so with one slot this continues the
  // operation's own booking.
  if (Cycles != 0)
    Outstanding.reserve(From, Cycles);
}

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

ExtMemModel::ExtMemModel(uint32_t Latency, double BytesPerCycle)
    : Latency(Latency), BytesPerCycle(BytesPerCycle), Link("ext_mem_link", 1) {
  if (BytesPerCycle < 0)
    throw std::invalid_argument("ext_mem_bytes_per_cycle must be at least 0");
}

ExtMemModel::Slot ExtMemModel::transfer(uint64_t Bytes, Cycle Earliest) {
  if (!isTimed())
    return {Earliest, Earliest};
  const auto DataCycles = static_cast<uint64_t>(
      std::ceil(static_cast<double>(Bytes) / BytesPerCycle));
  const uint64_t Occupancy = Latency + std::max<uint64_t>(DataCycles, 1);
  const Cycle Start = Link.reserve(Earliest, Occupancy);
  return {Start + Latency, Start + Occupancy};
}

Cycle ExtMemModel::getReady(const Slot &Booked, uint64_t Bytes) const {
  if (!isTimed())
    return Booked.FirstByte;
  const auto DataCycles = static_cast<uint64_t>(
      std::ceil(static_cast<double>(Bytes) / BytesPerCycle));
  return std::min(Booked.Done, Booked.FirstByte + std::max<uint64_t>(DataCycles, 1));
}
