//===- Lsu.cpp - The load/store unit --------------------------------------===//

#include "core/Lsu.h"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <cstdint>

using namespace reef_perf;

Lsu::Lsu(sparta::TreeNode *Node, const LsuParameterSet *Params)
    : sparta::Unit(Node, name), QueueEntries(Params->lsu_queue_entries),
      Slot("lsu", 1) {
  InInstsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Lsu, receiveInst, InstRef));
  sparta::StartupEvent(Node, CREATE_SPARTA_HANDLER(Lsu, sendInitialCredits));
}

void Lsu::sendInitialCredits() { OutCreditsPort.send(QueueEntries); }

void Lsu::receiveInst(const InstRef &I) {
  sparta_assert(Mem, "memory was not connected to the LSU");
  Cycle Now = getClock()->currentCycle();
  ++NumExecuted;

  MemRequest Request;
  Request.Requester = RequesterKind::LSU;
  Request.Accesses = I->Mem;
  Request.Earliest = Slot.getNextFree(Now);
  MemResponse Response = Mem->access(Request);

  Cycle Start = Slot.reserve(Response.Start, Response.Occupancy);
  sparta_assert(Start == Response.Start, "LSU slot busy at the memory's start");
  I->IssueCycle = Start;
  I->ResultReadyCycle = Start + Response.Latency - 1;
  I->CompleteCycle = Start + Response.Latency;

  // The instruction leaves the queue when it starts.
  OutCreditsPort.send(1, Start - Now);
}
