//===- Dispatch.cpp - The dispatch unit -----------------------------------===//

#include "backend/Dispatch.h"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

using namespace reef_perf;

const char *reef_perf::getStallReasonName(StallReason Reason) {
  switch (Reason) {
  case StallReason::IbufEmpty:
    return "ibuf_empty";
  case StallReason::Raw:
    return "raw_hazard";
  case StallReason::Waw:
    return "waw_hazard";
  case StallReason::RobFull:
    return "rob_full";
  case StallReason::LsuFull:
    return "lsu_queue_full";
  case StallReason::VecFull:
    return "vector_queue_full";
  case StallReason::MtxFull:
    return "matrix_queue_full";
  case StallReason::NumReasons:
    break;
  }
  return "unknown";
}

Dispatch::Dispatch(sparta::TreeNode *Node, const DispatchParameterSet *Params)
    : sparta::Unit(Node, name), DispatchWidth(Params->dispatch_width),
      IbufEntries(Params->ibuf_entries) {
  sparta_assert(DispatchWidth > 0 && IbufEntries > 0);

  constexpr auto NumReasons = static_cast<size_t>(StallReason::NumReasons);
  for (size_t Idx = 0; Idx < NumReasons; ++Idx) {
    const char *ReasonName = getStallReasonName(static_cast<StallReason>(Idx));
    StallCounters.push_back(std::make_unique<sparta::Counter>(
        &unit_stat_set_, std::string("stall_") + ReasonName,
        std::string("Cycles with no dispatch, blamed on: ") + ReasonName,
        sparta::Counter::COUNT_NORMAL));
  }

  InInstsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Dispatch, receiveInsts, FetchPacket));
  InRobCreditsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Dispatch, receiveRobCredits, uint32_t));
  InLsuCreditsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Dispatch, receiveLsuCredits, uint32_t));
  InVecCreditsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Dispatch, receiveVecCredits, uint32_t));
  InMtxCreditsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Dispatch, receiveMtxCredits, uint32_t));
  sparta::StartupEvent(Node,
                       CREATE_SPARTA_HANDLER(Dispatch, sendInitialCredits));
}

void Dispatch::sendInitialCredits() { OutFetchCreditsPort.send(IbufEntries); }

void Dispatch::receiveInsts(const FetchPacket &Packet) {
  for (const InstRef &I : Packet.Insts)
    Ibuf.push_back(I);
  sparta_assert(Ibuf.size() <= IbufEntries, "instruction buffer overflow");
  if (Packet.Last)
    FetchDone = true;
  scheduleDispatch();
}

void Dispatch::receiveRobCredits(const uint32_t &Credits) {
  RobCredits += Credits;
  scheduleDispatch();
}

void Dispatch::receiveLsuCredits(const uint32_t &Credits) {
  LsuCredits += Credits;
  scheduleDispatch();
}

void Dispatch::receiveVecCredits(const uint32_t &Credits) {
  VecCredits += Credits;
  scheduleDispatch();
}

void Dispatch::receiveMtxCredits(const uint32_t &Credits) {
  MtxCredits += Credits;
  scheduleDispatch();
}

void Dispatch::scheduleDispatch() {
  // Keep ticking every cycle until the program is fully dispatched, so cycles
  // with an empty buffer are counted as IbufEmpty stalls.
  if (!Ibuf.empty() || !FetchDone)
    DispatchEvent.schedule(); // This cycle.
}

bool Dispatch::canDispatch(const InstRef &I, uint64_t Now,
                           StallReason &Why) const {
  for (uint16_t Reg : I->Srcs) {
    const InstRef &Writer = LastWriter.at(Reg);
    if (Writer && Writer->ResultReadyCycle > Now) {
      Why = StallReason::Raw;
      return false;
    }
  }
  for (uint16_t Reg : I->Dsts) {
    const InstRef &Writer = LastWriter.at(Reg);
    if (Writer && Writer->ResultReadyCycle > Now) {
      Why = StallReason::Waw;
      return false;
    }
  }
  if (RobCredits == 0) {
    Why = StallReason::RobFull;
    return false;
  }
  uint32_t QueueCredits = 0;
  StallReason QueueFull = StallReason::NumReasons;
  switch (getExecTarget(*I)) {
  case ExecTarget::Scalar:
    return true; // The scalar units have no queue.
  case ExecTarget::LSU:
    QueueCredits = LsuCredits;
    QueueFull = StallReason::LsuFull;
    break;
  case ExecTarget::Vector:
    QueueCredits = VecCredits;
    QueueFull = StallReason::VecFull;
    break;
  case ExecTarget::Matrix:
    QueueCredits = MtxCredits;
    QueueFull = StallReason::MtxFull;
    break;
  }
  if (QueueCredits == 0) {
    Why = QueueFull;
    return false;
  }
  return true;
}

void Dispatch::sendToTarget(const InstRef &I) {
  switch (getExecTarget(*I)) {
  case ExecTarget::Scalar:
    OutScalarPort.send(I);
    break;
  case ExecTarget::LSU:
    --LsuCredits;
    OutLsuPort.send(I);
    break;
  case ExecTarget::Vector:
    --VecCredits;
    OutVectorPort.send(I);
    break;
  case ExecTarget::Matrix:
    --MtxCredits;
    OutMatrixPort.send(I);
    break;
  }
}

void Dispatch::dispatchGroup() {
  uint64_t Now = getClock()->currentCycle();
  uint32_t Count = 0;
  StallReason Why = StallReason::IbufEmpty;

  while (Count < DispatchWidth && !Ibuf.empty()) {
    InstRef I = Ibuf.front();
    if (!canDispatch(I, Now, Why))
      break;
    I->DispatchCycle = Now;
    for (uint16_t Reg : I->Dsts)
      LastWriter.at(Reg) = I;
    --RobCredits;
    sendToTarget(I);
    OutRobPort.send(I);
    Ibuf.pop_front();
    ++Count;
  }

  if (Count > 0) {
    NumDispatched += Count;
    ++CyclesWithDispatch;
    OutFetchCreditsPort.send(Count);
  } else {
    ++(*StallCounters.at(static_cast<size_t>(Why)));
  }

  if (!Ibuf.empty() || !FetchDone)
    DispatchEvent.schedule(1);
}
