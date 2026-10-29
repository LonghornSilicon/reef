//===- Lsu.cpp - The load/store unit --------------------------------------===//

#include "core/Lsu.h"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <cstdint>

using namespace reef_perf;

Lsu::Lsu(sparta::TreeNode *Node, const LsuParameterSet *Params)
    : sparta::Unit(Node, name), QueueEntries(Params->lsu_queue_entries),
      ScalarLatencyOverhead(Params->lsu_scalar_latency_overhead),
      ScalarOccupancyOverhead(Params->lsu_scalar_occupancy_overhead),
      VectorLoadRegOverhead(Params->lsu_vector_load_reg_overhead),
      VectorStoreRegOverhead(Params->lsu_vector_store_reg_overhead),
      VectorSwitchCycles(Params->lsu_vector_switch_cycles), Pipe("lsu", 1) {
  InInstsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Lsu, receiveInst, InstRef));
  sparta::StartupEvent(Node, CREATE_SPARTA_HANDLER(Lsu, sendInitialCredits));
}

void Lsu::sendInitialCredits() { OutCreditsPort.send(QueueEntries); }

void Lsu::receiveInst(const InstRef &I) {
  sparta_assert(Mem, "memory was not connected to the LSU");
  sparta_assert(!I->isVector(), "vector memory instructions come through the "
                                "vector module");
  Cycle Now = getClock()->currentCycle();
  ++NumExecuted;

  MemRequest Request;
  Request.Requester = RequesterKind::LSU;
  Request.Accesses = I->Mem;
  Request.Earliest = Pipe.getNextFree(Now);
  MemResponse Response = Mem->access(Request);
  Cycle Start = Response.Start;
  // Memory reports the physical transfer. Scalar instructions also traverse
  // LSU pipeline stages; vector registers add their path delay separately in
  // accessVectorRegister().
  Cycle Reserved =
      Pipe.reserve(Start, Response.Occupancy + ScalarOccupancyOverhead);
  sparta_assert(Reserved == Start, "LSU busy at the memory's start");
  I->IssueCycle = Start;
  I->ResultReadyCycle = Start + Response.Latency + ScalarLatencyOverhead - 1;
  I->CompleteCycle = Start + Response.Latency + ScalarLatencyOverhead;

  // The instruction leaves the queue when it starts.
  OutCreditsPort.send(1, Start - Now);
}

VectorLsuResponse Lsu::accessVectorRegister(const VectorLsuRequest &Request) {
  sparta_assert(Mem, "memory was not connected to the LSU");
  ++NumVectorRegisters;
  uint64_t Overhead =
      Request.IsStore ? VectorStoreRegOverhead : VectorLoadRegOverhead;
  MemRequest Access;
  Access.Requester = RequesterKind::LSU;
  Access.Accesses = Request.Accesses;
  Access.Earliest = Pipe.getNextFree(Request.Earliest) + Overhead;
  MemResponse Response = Mem->access(Access);

  VectorLsuResponse Result;
  Result.Start = Response.Start - Overhead;
  Cycle BusyUntil = Response.Start + Response.Occupancy;
  if (Request.LastRegister)
    BusyUntil += VectorSwitchCycles;
  Cycle Reserved = Pipe.reserve(Result.Start, BusyUntil - Result.Start);
  sparta_assert(Reserved == Result.Start, "LSU busy at the memory's start");
  // Dispatch gave the instruction an LSU queue entry; it leaves the queue when
  // its first register starts.
  if (Request.FirstRegister) {
    Cycle Now = getClock()->currentCycle();
    sparta_assert(Result.Start >= Now);
    OutCreditsPort.send(1, Result.Start - Now);
  }
  // The data is usable at Start + Latency; the LSU hands it over two cycles
  // earlier (RTL waveform of vec_add_m1: a load is taken in A, its data is
  // handed back in A + 3 and usable in A + 5).
  Result.Done = Response.Start + Response.Latency - 2;
  return Result;
}
