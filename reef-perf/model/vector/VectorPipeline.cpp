//===- VectorPipeline.cpp - The vector backend pipeline -------------------===//

#include "vector/VectorPipeline.h"

#include "sparta/utils/SpartaAssert.hpp"

#include <algorithm>
#include <cstdint>

using namespace reef_perf;

InOrderSlots::InOrderSlots(uint32_t Width) : Width(Width) {
  sparta_assert(Width > 0);
}

Cycle InOrderSlots::take(Cycle Earliest) {
  if (Earliest > Current) {
    Current = Earliest;
    Used = 0;
  }
  if (Used == Width) {
    ++Current;
    Used = 0;
  }
  ++Used;
  return Current;
}

QueueOccupancy::QueueOccupancy(uint32_t Entries, Cycle RefillDelay)
    : Entries(Entries), RefillDelay(RefillDelay) {
  sparta_assert(Entries > 0);
}

Cycle QueueOccupancy::getFreeCycle() const {
  return Leaves.size() < Entries ? 0 : Leaves.front() + RefillDelay;
}

void QueueOccupancy::recordLeave(Cycle Leave) {
  Leaves.push_back(Leave);
  if (Leaves.size() > Entries)
    Leaves.pop_front();
}

VectorPipeline::VectorPipeline(const VectorPipelineConfig &Config)
    : De1Slots(Config.De1Width), De2Slots(Config.De2Width),
      De2UopSlots(Config.De2UopWidth), DispatchSlots(Config.DispatchWidth),
      RetireSlots(Config.RetireWidth), LegalQueue(Config.LegalQueueEntries),
      UopQueue(Config.UopQueueEntries), Rob(Config.RobEntries, 0) {
  for (uint32_t Entries : Config.StationEntries)
    Stations.emplace_back(Entries);
  LsuStation = Stations.size();
  Stations.emplace_back(Config.LsuStationEntries);
  LastStart.assign(Stations.size(), 0);
  for (const VectorPoolConfig &Pool : Config.Pools) {
    sparta_assert(Pool.Station < Stations.size());
    Pools.emplace_back(Pool.Name, Pool.Count);
    PoolStations.push_back(Pool.Station);
  }
}

std::vector<const ResourcePool *> VectorPipeline::getPools() const {
  std::vector<const ResourcePool *> Result;
  for (const ResourcePool &Pool : Pools)
    Result.push_back(&Pool);
  return Result;
}

VectorTiming VectorPipeline::schedule(Cycle Dispatched, const VectorOp &Op,
                                      const std::vector<Cycle> &SourcesReady) {
  bool UsesLsu = static_cast<bool>(Op.BookLsu);
  sparta_assert(Op.Uops > 0 && Op.Latency > 0);
  sparta_assert(SourcesReady.empty() || SourcesReady.size() == Op.Uops);
  sparta_assert(UsesLsu || Op.Pool < Pools.size());
  size_t StationIndex = UsesLsu ? LsuStation : PoolStations[Op.Pool];
  QueueOccupancy &Station = Stations[StationIndex];
  VectorTiming T;
  // The frontend's inst_q holds it in D+1 and pushes it into the command
  // queue, so DE1 sees it in D+2, once the legal command queue has room.
  T.CommandQueuePop =
      De1Slots.take(std::max(Dispatched + 2, LegalQueue.getFreeCycle()));

  // DE2 splits it into uops the next cycle, as uop-queue space allows; it
  // leaves the legal command queue with its last uop.
  Cycle Decode = De2Slots.take(std::max(T.CommandQueuePop + 1, LastUopPush));
  for (uint32_t Uop = 0; Uop < Op.Uops; ++Uop) {
    Cycle Push = De2UopSlots.take(std::max(Decode, UopQueue.getFreeCycle()));
    LastUopPush = Push;
    // Compares and reductions write one result, from their last uop, and only
    // that uop takes a ROB entry (rvv_backend_decode_unit_ari_de2.sv).
    bool TakesRobEntry = !Op.OneRobEntry || Uop + 1 == Op.Uops;

    // Dispatch needs a vector ROB entry and a reservation-station entry, and
    // its sources: a dependent uop dispatches in the cycle its producer
    // retires, reading the result from the ROB, and starts the next cycle.
    Cycle Ready = SourcesReady.empty() ? 0 : SourcesReady[Uop];
    Cycle Dispatch = DispatchSlots.take(
        std::max({Push + 1, TakesRobEntry ? Rob.getFreeCycle() : 0,
                  Station.getFreeCycle(), Ready > 0 ? Ready - 1 : 0}));
    UopQueue.recordLeave(Dispatch);

    Cycle Earliest = std::max(Dispatch + 1, LastStart[StationIndex]);
    Cycle Start = 0;
    Cycle Done = 0; // The result leaves the unit, or comes back from the LSU.
    if (UsesLsu) {
      VectorLsuResponse Lsu = Op.BookLsu(Uop, Earliest);
      Start = Lsu.Start;
      Done = Lsu.Done;
    } else {
      ResourcePool &Units = Pools[Op.Pool];
      Start = Op.FirstUnitOnly ? Units.reserveFirst(Earliest, Op.Occupancy)
                               : Units.reserve(Earliest, Op.Occupancy);
      Done = Start + Op.Latency - 1;
    }
    LastStart[StationIndex] = Start;
    Station.recordLeave(Start);
    if (Uop == 0) {
      T.FirstStart = Start;
      T.FirstDone = Done;
    }
    if (!TakesRobEntry) {
      T.UopRetires.push_back(0); // Set below, to the instruction's retire.
      continue;
    }

    // A cycle in the result FIFO, then the ROB marks the uop done and can
    // retire it, in order.
    Cycle Retire = RetireSlots.take(std::max(Done + 2, LastRetire));
    LastRetire = Retire;
    Rob.recordLeave(Retire);
    T.Retire = Retire;
    T.UopRetires.push_back(Retire);
  }
  if (Op.OneRobEntry)
    std::ranges::fill(T.UopRetires, T.Retire);
  LegalQueue.recordLeave(LastUopPush);
  return T;
}
