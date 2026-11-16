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

/// \returns true for a Zvt tile load or store, which CoralNPU sends to the
/// LSU as well as to the RVV backend at dispatch (Decode.scala, VTLOAD and
/// VTSTORE).
static bool isMatrixMemory(const Inst &I) {
  return I.Mtx.Op == MatrixOp::TileLoad || I.Mtx.Op == MatrixOp::TileStore;
}

/// \returns true if Dispatch tracks \p Reg, last written by \p Writer.
/// The vector module tracks vector registers that vector instructions
/// write; Dispatch tracks the ones the matrix module writes (vtmv.v.t) only
/// for WAW, see isMatrixVectorRaw for RAW.
static bool isTracked(RegId Reg, const InstRef &Writer) {
  return Writer && (Reg.getFile() != RegFile::V || Writer->isMatrix());
}

/// \returns true if \p Reg is a vector register last written by a matrix
/// instruction. Its readers wait for the result inside the vector module
/// (the matrix module records it with setRegisterReady, as CoralNPU's RVV
/// backend resolves the dependence); Dispatch only keeps them out of the
/// writer's own cycle, when the vector module would book them before the
/// matrix module records the write.
static bool isMatrixVectorRaw(RegId Reg, const InstRef &Writer) {
  return Writer && Reg.getFile() == RegFile::V && Writer->isMatrix();
}

/// \returns true if \p I reads \p Reg from the register file's flops, without
/// write forwarding: a load or store address (rs1), scalar store data (rs2) or
/// a jalr target (Decode.scala:347-361, usesRs1Regd and usesRs2Regd).
static bool readsRegistered(const Inst &I, RegId Reg) {
  if (Reg.getFile() != RegFile::X)
    return false;
  const unsigned Rs1 = (I.Encoding >> 15) & 31;
  const unsigned Rs2 = (I.Encoding >> 20) & 31;
  const bool IsJalr = (I.Encoding & 0x7f) == 0x67;
  if ((I.isMemory() || IsJalr) && Reg.getIndex() == Rs1)
    return true;
  return I.Class == InstClass::STORE && Reg.getIndex() == Rs2;
}

/// \returns true if \p I must dispatch first in its cycle with nothing else
/// (Decode.scala, forceSlot0Only): CSR, fence, system and FP instructions,
/// a vector instruction that reads or writes an f register, and vtdiscard.
static bool isSlot0Only(const Inst &I) {
  switch (I.Class) {
  case InstClass::CSR:
  case InstClass::FENCE:
  case InstClass::SYSTEM:
  case InstClass::FP:
  case InstClass::FP_DIV:
  case InstClass::FP_LOAD:
  case InstClass::FP_STORE:
    return true;
  default:
    break;
  }
  if (I.Mtx.Op == MatrixOp::Discard)
    return true;
  if (!I.isVector())
    return false;
  auto IsF = [](RegId Reg) { return Reg.getFile() == RegFile::F; };
  return std::ranges::any_of(I.Srcs, IsF) || std::ranges::any_of(I.Dsts, IsF);
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
  case StallReason::CsrDrain:
    return "csr_drain";
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
    : sparta::Unit(Node, name), DispatchWidth(Params->dispatch_width),
      LoadWawAdvance(Params->load_waw_advance),
      VectorScalarWawDelay(Params->vector_scalar_waw_delay),
      BranchEndsGroup(Params->branch_ends_group),
      Slot0OnlyRule(Params->slot0_only_rule),
      JumpEndsGroup(Params->jump_ends_group),
      CsrWaitsForEmptyRob(Params->csr_waits_for_empty_rob),
      VtdiscardInterlock(Params->vtdiscard_interlock),
      RvvConfigInterlock(Params->rvv_config_interlock) {
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
  if (RobSize == 0)
    RobSize = Credits;
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
  // A CSR instruction, and vtdiscard, wait for an empty Rob. Every credit
  // back means it is empty; they return the cycle after the last retire, as
  // the RTL's registered count does.
  bool Drains = (CsrWaitsForEmptyRob && I->Class == InstClass::CSR) ||
                (VtdiscardInterlock && I->Mtx.Op == MatrixOp::Discard);
  if (Drains && RobCredits < RobSize) {
    Why = StallReason::CsrDrain;
    InstRef Oldest = TheRob ? TheRob->getOldest() : nullptr;
    Blamed = Oldest ? getExecTarget(*Oldest) : ExecTarget::Scalar;
    return false;
  }
  // The RVV front end holds each instruction for a cycle, and its config
  // state is not valid for a load or store meanwhile.
  if (RvvConfigInterlock && (isVectorMemory(*I) || isMatrixMemory(*I)) &&
      (LastRvvDispatch + 1 == Now || LastVsetDispatch == Now)) {
    Why = StallReason::VecFull;
    Blamed = ExecTarget::Vector;
    return false;
  }
  // A register hazard is blamed on the unit producing the register. A
  // registered read sees the value one cycle after a forwarded read. A
  // writer not scheduled yet (Never) blocks as is: Never + 1 would wrap to 0.
  for (RegId Reg : I->Srcs) {
    const InstRef &Writer = LastWriter.at(Reg.getDense());
    bool Waits;
    if (isMatrixVectorRaw(Reg, Writer)) {
      Waits = Writer->DispatchCycle >= Now;
    } else if (!isTracked(Reg, Writer)) {
      Waits = false;
    } else {
      Cycle Ready = Writer->ResultReadyCycle;
      if (Ready != Inst::Never && readsRegistered(*I, Reg))
        ++Ready;
      Waits = Ready > Now;
    }
    if (Waits) {
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
    const Cycle WawAdvance =
        Writer && Writer->Class == InstClass::LOAD ? LoadWawAdvance : 0;
    // Two matrix writers of one vector register (vtmv.v.t after vtmv.v.t)
    // need no wait: the matrix module runs them in order.
    if (isMatrixVectorRaw(Reg, Writer) && I->isMatrix())
      continue;
    // A vector instruction that writes an x register last written by
    // another vector instruction waits a cycle more than a scalar writer.
    const Cycle WawDelay = Writer && Writer->isVector() && I->isVector() &&
                                   Reg.getFile() == RegFile::X
                               ? VectorScalarWawDelay
                               : 0;
    if (isTracked(Reg, Writer) &&
        (Writer->ResultReadyCycle == Inst::Never ||
         Writer->ResultReadyCycle + WawDelay > Now + WawAdvance)) {
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
  if ((isVectorMemory(*I) || isMatrixMemory(*I)) && LsuCredits == 0) {
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
    if (isMatrixMemory(*I))
      --LsuCredits; // Freed by the LSU when the first beat starts.
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
    if (!I)
      break;
    bool Alone = Slot0OnlyRule && isSlot0Only(*I);
    if (Alone && Count > 0)
      break;
    if (!canDispatch(I, Now, Why, Blamed))
      break;
    I->DispatchCycle = Now;
    if (I->isVector() || I->isMatrix())
      LastRvvDispatch = Now;
    if (I->Class == InstClass::VSET)
      LastVsetDispatch = Now;
    for (RegId Reg : I->Dsts)
      LastWriter.at(Reg.getDense()) = I;
    --RobCredits;
    sendToTarget(I);
    OutRobPort.send(I);
    Frontend->pop(Now);
    ++Count;
    if (Alone || (BranchEndsGroup && I->Class == InstClass::BRANCH))
      break;
    if (JumpEndsGroup &&
        (I->Class == InstClass::JUMP || I->Class == InstClass::SYSTEM))
      break;
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
