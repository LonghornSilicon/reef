//===- Rob.cpp - The retirement buffer ------------------------------------===//

#include "backend/Rob.h"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <cstdint>

using namespace reef_perf;

Rob::Rob(sparta::TreeNode *Node, const RobParameterSet *Params)
    : sparta::Unit(Node, name), RobEntries(Params->rob_entries),
      RetireWidth(Params->retire_width) {
  sparta_assert(RobEntries > 0 && RetireWidth > 0);
  InInstsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Rob, receiveInst, InstRef));
  sparta::StartupEvent(Node, CREATE_SPARTA_HANDLER(Rob, sendInitialCredits));
}

void Rob::sendInitialCredits() { OutCreditsPort.send(RobEntries); }

void Rob::receiveInst(const InstRef &I) {
  Entries.push_back(I);
  sparta_assert(Entries.size() <= RobEntries, "Rob overflow");
  RetireEvent.schedule(); // This cycle.
}

void Rob::retireInsts() {
  uint64_t Now = getClock()->currentCycle();
  uint32_t Count = 0;
  while (Count < RetireWidth && !Entries.empty() &&
         Entries.front()->CompleteCycle <= Now) {
    Entries.front()->RetireCycle = Now;
    LastRetireCycle = Now;
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
  uint64_t HeadDone = Entries.front()->CompleteCycle;
  if (HeadDone == Inst::Never || HeadDone <= Now)
    RetireEvent.schedule(1);
  else
    RetireEvent.schedule(HeadDone - Now);
}
