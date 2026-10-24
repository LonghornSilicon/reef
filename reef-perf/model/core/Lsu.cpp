//===- Lsu.cpp - The load/store unit --------------------------------------===//

#include "core/Lsu.h"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

using namespace reef_perf;

std::vector<std::span<const MemAccess>>
reef_perf::splitByRegister(std::span<const MemAccess> Accesses,
                           uint32_t RegBytes) {
  if (RegBytes == 0)
    throw std::invalid_argument("register size must be at least 1 byte");
  std::vector<std::span<const MemAccess>> Groups;
  size_t Begin = 0;
  uint32_t Bytes = 0;
  for (size_t Idx = 0; Idx < Accesses.size(); ++Idx) {
    Bytes += Accesses[Idx].Size;
    if (Bytes >= RegBytes || Idx + 1 == Accesses.size()) {
      Groups.push_back(Accesses.subspan(Begin, Idx + 1 - Begin));
      Begin = Idx + 1;
      Bytes = 0;
    }
  }
  return Groups;
}

Lsu::Lsu(sparta::TreeNode *Node, const LsuParameterSet *Params)
    : sparta::Unit(Node, name), QueueEntries(Params->lsu_queue_entries),
      VectorLoadRegOverhead(Params->lsu_vector_load_reg_overhead),
      VectorStoreRegOverhead(Params->lsu_vector_store_reg_overhead),
      Pipe("lsu", 1) {
  InInstsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Lsu, receiveInst, InstRef));
  sparta::StartupEvent(Node, CREATE_SPARTA_HANDLER(Lsu, sendInitialCredits));
}

void Lsu::sendInitialCredits() { OutCreditsPort.send(QueueEntries); }

void Lsu::receiveInst(const InstRef &I) {
  sparta_assert(Mem, "memory was not connected to the LSU");
  Cycle Now = getClock()->currentCycle();
  ++NumExecuted;

  // A vector access passes through the slot one register at a time, each
  // pass with its own overhead and memory request. Anything else is one pass
  // with no overhead.
  bool IsVector =
      I->Class == InstClass::V_LOAD || I->Class == InstClass::V_STORE;
  std::vector<std::span<const MemAccess>> Passes;
  if (IsVector && !I->Mem.empty())
    Passes = splitByRegister(I->Mem, SlotBytes);
  else
    Passes.emplace_back(I->Mem);
  uint64_t Overhead = 0;
  if (I->Class == InstClass::V_LOAD)
    Overhead = VectorLoadRegOverhead;
  else if (I->Class == InstClass::V_STORE)
    Overhead = VectorStoreRegOverhead;

  Cycle Start = 0;
  // A vector store or masked access waits in the slot for its vector
  // sources; nothing behind it can go first.
  Cycle BusyUntil = Pipe.getNextFree(std::max(Now, I->VecSourcesReadyCycle));
  Cycle Ready = 0;
  for (size_t Idx = 0; Idx < Passes.size(); ++Idx) {
    MemRequest Request;
    Request.Requester = RequesterKind::LSU;
    Request.Accesses = Passes[Idx];
    Request.Earliest = BusyUntil + Overhead;
    MemResponse Response = Mem->access(Request);
    if (Idx == 0)
      Start = Response.Start - Overhead;
    BusyUntil = Response.Start + Response.Occupancy;
    Ready = Response.Start + Response.Latency;
  }

  Cycle Reserved = Pipe.reserve(Start, BusyUntil - Start);
  sparta_assert(Reserved == Start, "LSU busy at the memory's start");
  I->IssueCycle = Start;
  I->ResultReadyCycle = Ready - 1;
  I->CompleteCycle = Ready;

  // The instruction leaves the queue when it starts.
  OutCreditsPort.send(1, Start - Now);
}
