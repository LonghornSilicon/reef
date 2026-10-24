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

QueueOccupancy::QueueOccupancy(uint32_t Entries) : Entries(Entries) {
  sparta_assert(Entries > 0);
}

Cycle QueueOccupancy::getFreeCycle() const {
  return Leaves.size() < Entries ? 0 : Leaves.front() + 1;
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
      UopQueue(Config.UopQueueEntries), Station(Config.StationEntries),
      Rob(Config.RobEntries) {}

VectorTiming VectorPipeline::schedule(Cycle Dispatched, uint32_t Uops,
                                      uint64_t Latency, uint64_t Occupancy,
                                      Cycle SourcesReady, ResourcePool &Units) {
  sparta_assert(Uops > 0 && Latency > 0);
  VectorTiming T;
  // The frontend's inst_q holds it in D+1 and pushes it into the command
  // queue, so DE1 sees it in D+2, once the legal command queue has room.
  T.CommandQueuePop =
      De1Slots.take(std::max(Dispatched + 2, LegalQueue.getFreeCycle()));

  // DE2 splits it into uops the next cycle, as uop-queue space allows; it
  // leaves the legal command queue with its last uop.
  Cycle Decode = De2Slots.take(std::max(T.CommandQueuePop + 1, LastUopPush));
  for (uint32_t Uop = 0; Uop < Uops; ++Uop) {
    Cycle Push = De2UopSlots.take(std::max(Decode, UopQueue.getFreeCycle()));
    LastUopPush = Push;

    // Dispatch needs a vector ROB entry and a reservation-station entry, and
    // its sources: a dependent uop dispatches in the cycle its producer
    // retires, reading the result from the ROB, and starts the next cycle.
    Cycle Dispatch = DispatchSlots.take(
        std::max({Push + 1, Rob.getFreeCycle(), Station.getFreeCycle(),
                  SourcesReady > 0 ? SourcesReady - 1 : 0}));
    UopQueue.recordLeave(Dispatch);

    Cycle Start = Units.reserve(std::max(Dispatch + 1, LastStart), Occupancy);
    LastStart = Start;
    Station.recordLeave(Start);
    if (Uop == 0)
      T.FirstStart = Start;

    // After the unit: a cycle in the result FIFO, then the ROB marks it done
    // and can retire it, in order.
    Cycle Retire = RetireSlots.take(std::max(Start + Latency + 1, LastRetire));
    LastRetire = Retire;
    Rob.recordLeave(Retire);
    T.Retire = Retire;
  }
  LegalQueue.recordLeave(LastUopPush);
  return T;
}
