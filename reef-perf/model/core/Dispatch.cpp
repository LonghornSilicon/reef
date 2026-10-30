//===- Dispatch.cpp - The dispatch unit -----------------------------------===//

#include "core/Dispatch.h"
#include "core/Rob.h"

#include "sparta/utils/SpartaAssert.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

using namespace reef_perf;

/// \returns true for a vector load or store.
static bool isVectorMemory(const Inst &I) {
  return I.Class == InstClass::V_LOAD || I.Class == InstClass::V_STORE;
}

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

const char *reef_perf::getBoundName(ExecTarget Target) {
  switch (Target) {
  case ExecTarget::Scalar:
    return "core";
  case ExecTarget::LSU:
    return "memory";
  case ExecTarget::Vector:
    return "vector";
  case ExecTarget::Matrix:
    return "matrix";
  }
  return "unknown";
}

uint64_t TopDownSlots::getTotal() const {
  uint64_t Total = Retiring;
  for (uint64_t N : Frontend)
    Total += N;
  for (const auto &ByReason : Backend)
    for (uint64_t N : ByReason)
      Total += N;
  return Total;
}

Dispatch::Dispatch(sparta::TreeNode *Node, const DispatchParameterSet *Params)
    : sparta::Unit(Node, name), DispatchWidth(Params->dispatch_width) {
  sparta_assert(DispatchWidth > 0);
  Slots.Width = DispatchWidth;

  constexpr auto NumReasons = static_cast<size_t>(StallReason::NumReasons);
  for (size_t Idx = 0; Idx < NumReasons; ++Idx) {
    const char *ReasonName = getStallReasonName(static_cast<StallReason>(Idx));
    StallCounters.push_back(std::make_unique<sparta::Counter>(
        &unit_stat_set_, std::string("stall_") + ReasonName,
        std::string("Cycles with no dispatch, blamed on: ") + ReasonName,
        sparta::Counter::COUNT_NORMAL));
  }

  InRobCreditsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Dispatch, receiveRobCredits, uint32_t));
  InLsuCreditsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Dispatch, receiveLsuCredits, uint32_t));
  InVecCreditsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Dispatch, receiveVecCredits, uint32_t));
  InMtxCreditsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Dispatch, receiveMtxCredits, uint32_t));
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
  // with nothing to take are counted as IbufEmpty stalls.
  if (!Frontend->isDrained(getClock()->currentCycle()))
    DispatchEvent.schedule(); // This cycle.
}

bool Dispatch::canDispatch(const InstRef &I, Cycle Now, StallReason &Why,
                           ExecTarget &Blamed) const {
  // A register hazard is blamed on the unit producing the register.
  for (RegId Reg : I->Srcs) {
    const InstRef &Writer = LastWriter.at(Reg.getDense());
    if (Reg.getFile() != RegFile::V && Writer &&
        Writer->ResultReadyCycle > Now) {
      Why = StallReason::Raw;
      Blamed = getExecTarget(*Writer);
      return false;
    }
  }
  for (RegId Reg : I->Dsts) {
    const InstRef &Writer = LastWriter.at(Reg.getDense());
    // M3's combinational scoreboard clear lets a following writer dispatch
    // one cycle before a load result is ready. RAW dependencies still wait
    // for ResultReadyCycle.
    const Cycle WawAdvance = Writer && Writer->Class == InstClass::LOAD ? 1 : 0;
    if (Reg.getFile() != RegFile::V && Writer &&
        Writer->ResultReadyCycle > Now + WawAdvance) {
      Why = StallReason::Waw;
      Blamed = getExecTarget(*Writer);
      return false;
    }
  }
  // A full Rob is blamed on the unit its oldest instruction waits for.
  if (RobCredits == 0) {
    Why = StallReason::RobFull;
    InstRef Oldest = TheRob ? TheRob->getOldest() : nullptr;
    Blamed = Oldest ? getExecTarget(*Oldest) : ExecTarget::Scalar;
    return false;
  }
  uint32_t QueueCredits = 0;
  StallReason QueueFull = StallReason::NumReasons;
  switch (getExecTarget(*I)) {
  case ExecTarget::Scalar:
    return true; // The scalar functional units have no queue.
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
    Blamed = getExecTarget(*I);
    return false;
  }
  // A vector load or store also takes an LSU queue entry, which the LSU
  // frees when the instruction's first register starts (RTL waveform of
  // gemv_loads).
  if (isVectorMemory(*I) && LsuCredits == 0) {
    Why = StallReason::LsuFull;
    Blamed = ExecTarget::LSU;
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
    if (isVectorMemory(*I))
      --LsuCredits;
    OutVectorPort.send(I);
    break;
  case ExecTarget::Matrix:
    --MtxCredits;
    OutMatrixPort.send(I);
    break;
  }
}

void Dispatch::dispatchGroup() {
  Cycle Now = getClock()->currentCycle();
  uint32_t Count = 0;
  StallReason Why = StallReason::IbufEmpty;
  ExecTarget Blamed = ExecTarget::Scalar;

  while (Count < DispatchWidth) {
    InstRef I = Frontend->peek(Now);
    if (!I || !canDispatch(I, Now, Why, Blamed))
      break;
    I->DispatchCycle = Now;
    for (RegId Reg : I->Dsts)
      LastWriter.at(Reg.getDense()) = I;
    --RobCredits;
    sendToTarget(I);
    OutRobPort.send(I);
    Frontend->pop(Now);
    ++Count;
  }

  if (Count > 0) {
    NumDispatched += Count;
    ++CyclesWithDispatch;
  } else {
    ++(*StallCounters.at(static_cast<size_t>(Why)));
  }

  // The slots left over this cycle go to whatever stopped the group.
  Slots.Retiring += Count;
  uint32_t Lost = DispatchWidth - Count;
  if (Why == StallReason::IbufEmpty)
    Slots.Frontend.at(static_cast<size_t>(Frontend->getStall(Now))) += Lost;
  else
    Slots.Backend.at(static_cast<size_t>(Blamed))
        .at(static_cast<size_t>(Why)) += Lost;

  if (!Frontend->isDrained(Now))
    DispatchEvent.schedule(1);
}
