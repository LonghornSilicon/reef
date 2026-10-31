//===- Rob.cpp - The retirement buffer ------------------------------------===//

#include "core/Rob.h"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

using namespace reef_perf;

/// CSR number of mcycle.
static constexpr uint32_t McycleCsr = 0xB00;

Rob::Rob(sparta::TreeNode *Node, const RobParameterSet *Params)
    : sparta::Unit(Node, name), RobEntries(Params->rob_entries),
      RetireWidth(Params->retire_width),
      HeartbeatInterval(Params->heartbeat_interval) {
  sparta_assert(RobEntries > 0 && RetireWidth > 0);
  HeartbeatEvent.setContinuing(false);
  InInstsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Rob, receiveInst, InstRef));
  sparta::StartupEvent(Node, CREATE_SPARTA_HANDLER(Rob, sendInitialCredits));
}

void Rob::sendInitialCredits() {
  OutCreditsPort.send(RobEntries);
  if (HeartbeatInterval == 0)
    return;

  HeartbeatStart = std::chrono::steady_clock::now();
  LastHeartbeatTime = HeartbeatStart;
  HeartbeatEvent.schedule(HeartbeatInterval);
}

void Rob::printHeartbeat() {
  const Cycle Now = getClock()->currentCycle();
  const uint64_t Insts = NumRetired.get();
  const auto WallNow = std::chrono::steady_clock::now();

  const double Elapsed =
      std::chrono::duration<double>(WallNow - HeartbeatStart).count();
  const double IntervalSeconds =
      std::chrono::duration<double>(WallNow - LastHeartbeatTime).count();
  const Cycle IntervalCycles = Now - LastHeartbeatCycle;
  const uint64_t IntervalInsts = Insts - LastHeartbeatInsts;

  const double IntervalIPC =
      IntervalCycles == 0
          ? 0.0
          : static_cast<double>(IntervalInsts) /
                static_cast<double>(IntervalCycles);
  const double SimRate =
      Elapsed == 0.0 ? 0.0 : static_cast<double>(Now) / Elapsed;

  std::cerr << "[heartbeat] cycle=" << Now << " retired=" << Insts
            << " interval_ipc=" << std::fixed << std::setprecision(3)
            << IntervalIPC << " elapsed_s=" << Elapsed
            << " interval_s=" << IntervalSeconds
            << " sim_cycles_per_s=" << SimRate << std::endl;

  LastHeartbeatCycle = Now;
  LastHeartbeatInsts = Insts;
  LastHeartbeatTime = WallNow;
  HeartbeatEvent.schedule(HeartbeatInterval);
}

void Rob::receiveInst(const InstRef &I) {
  Entries.push_back(I);
  sparta_assert(Entries.size() <= RobEntries, "Rob overflow");
  RetireEvent.schedule(); // This cycle.
}

void Rob::retireInsts() {
  Cycle Now = getClock()->currentCycle();
  uint32_t Count = 0;
  while (Count < RetireWidth && !Entries.empty() &&
         Entries.front()->CompleteCycle <= Now) {
    const InstRef &Head = Entries.front();
    Head->RetireCycle = Now;
    LastRetireCycle = Now;
    if (Head->Class == InstClass::CSR && (Head->Encoding >> 20) == McycleCsr)
      McycleReads.push_back({Head->IssueCycle, Head->Seq});
    Entries.pop_front();
    ++Count;
  }
  if (Count > 0) {
    NumRetired += Count;
    OutCreditsPort.send(Count);
  }
  if (Entries.empty())
    return;

  // Sleep until the oldest instruction can complete. Poll every cycle while
  // it has not been timed yet, or when the retire width held it back.
  Cycle HeadDone = Entries.front()->CompleteCycle;
  if (HeadDone == Inst::Never || HeadDone <= Now)
    RetireEvent.schedule(1);
  else
    RetireEvent.schedule(HeadDone - Now);
}
