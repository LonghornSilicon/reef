//===- Fetch.cpp - The frontend module ------------------------------------===//

#include "frontend/Fetch.h"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <algorithm>
#include <cstdint>

using namespace reef_perf;

Fetch::Fetch(sparta::TreeNode *Node, const FetchParameterSet *Params)
    : sparta::Unit(Node, name), FetchWidth(Params->fetch_width),
      FetchInterval(Params->fetch_interval),
      RedirectPenalty(Params->redirect_penalty),
      IbufEntries(Params->ibuf_entries) {
  sparta_assert(FetchWidth > 0 && FetchInterval > 0 && IbufEntries > 0);
  // The first group is fetched in cycle 1.
  sparta::StartupEvent(Node, CREATE_SPARTA_HANDLER(Fetch, scheduleFirstFetch));
}

void Fetch::scheduleFirstFetch() { FetchEvent.schedule(1); }

uint32_t Fetch::getFreeEntries(Cycle Now) const {
  uint64_t Used = Ibuf.size() + (FreedCycle == Now ? FreedAtCycle : 0);
  return Used >= IbufEntries ? 0 : static_cast<uint32_t>(IbufEntries - Used);
}

InstRef Fetch::peek(Cycle Now) const {
  if (Ibuf.empty() || Ibuf.front()->FetchCycle >= Now)
    return nullptr;
  return Ibuf.front();
}

void Fetch::pop(Cycle Now) {
  sparta_assert(!Ibuf.empty(), "pop() from an empty instruction buffer");
  Ibuf.pop_front();
  if (FreedCycle != Now) {
    FreedCycle = Now;
    FreedAtCycle = 0;
  }
  ++FreedAtCycle;
  scheduleFetch(Now + 1); // The entry can be refilled next cycle.
}

bool Fetch::isDrained(Cycle Now) const {
  return Done && DoneCycle < Now && Ibuf.empty();
}

FrontendStall Fetch::getStall(Cycle /*Now*/) const {
  return LastGroupRedirected ? FrontendStall::BranchResteer
                             : FrontendStall::FetchBandwidth;
}

void Fetch::scheduleFetch(Cycle Earliest) {
  if (Done)
    return;
  Cycle Now = getClock()->currentCycle();
  FetchEvent.schedule(std::max(Earliest, NextFetchCycle) - Now);
}

void Fetch::fetchGroup() {
  sparta_assert(TheFuncSim, "FuncSim was not connected to Fetch");
  Cycle Now = getClock()->currentCycle();
  if (Done)
    return;
  if (Now < NextFetchCycle) { // Woken early by a pop(); wait.
    scheduleFetch(Now);
    return;
  }
  if (getFreeEntries(Now) == 0) {
    ++CyclesNoCredit;
    return; // pop() wakes us up.
  }

  uint32_t Fetched = 0;
  bool Redirect = false;
  while (Fetched < FetchWidth && getFreeEntries(Now) > 0) {
    InstRef I = TheFuncSim->next(); // Execute-at-fetch.
    if (!I) {
      Done = true;
      DoneCycle = Now;
      break;
    }
    I->FetchCycle = Now;
    Ibuf.push_back(I);
    ++Fetched;
    ++NumFetched;
    if (I->isRedirect()) {
      Redirect = true;
      ++NumRedirects;
      break; // The rest of this group would be on the wrong path.
    }
  }
  if (Fetched > 0) {
    ++NumGroups;
    LastGroupRedirected = Redirect;
  } else if (Done) {
    ++NumGroups;
  }
  NextFetchCycle = Now + FetchInterval + (Redirect ? RedirectPenalty : 0);
  if (getFreeEntries(Now) > 0)
    scheduleFetch(NextFetchCycle);
}
