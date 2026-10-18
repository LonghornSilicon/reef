//===- Fetch.cpp - The frontend module ------------------------------------===//

#include "frontend/Fetch.h"

#include "sparta/utils/SpartaAssert.hpp"

#include <cstdint>

using namespace reef_perf;

Fetch::Fetch(sparta::TreeNode *Node, const FetchParameterSet *Params)
    : sparta::Unit(Node, name), FetchWidth(Params->fetch_width),
      FetchInterval(Params->fetch_interval),
      RedirectPenalty(Params->redirect_penalty) {
  sparta_assert(FetchWidth > 0 && FetchInterval > 0);
  InCreditsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Fetch, receiveCredits, uint32_t));
  // Fetch starts when Dispatch sends its initial credits.
}

void Fetch::receiveCredits(const uint32_t &NewCredits) {
  Credits += NewCredits;
  scheduleFetch();
}

void Fetch::scheduleFetch() {
  if (Done || Credits == 0)
    return;
  uint64_t Now = getClock()->currentCycle();
  FetchEvent.schedule(NextFetchCycle > Now ? NextFetchCycle - Now : 0);
}

void Fetch::fetchGroup() {
  sparta_assert(TheFuncSim, "FuncSim was not connected to Fetch");
  uint64_t Now = getClock()->currentCycle();
  if (Done)
    return;
  if (Now < NextFetchCycle) { // Woken early by a credit; wait.
    scheduleFetch();
    return;
  }
  if (Credits == 0) {
    ++CyclesNoCredit;
    return; // receiveCredits() wakes us up.
  }

  FetchPacket Packet;
  bool Redirect = false;
  while (Packet.Insts.size() < FetchWidth && Credits > 0) {
    InstRef I = TheFuncSim->next(); // Execute-at-fetch.
    if (!I) {
      Done = true;
      Packet.Last = true;
      break;
    }
    I->FetchCycle = Now;
    Packet.Insts.push_back(I);
    --Credits;
    ++NumFetched;
    if (I->isRedirect()) {
      Redirect = true;
      ++NumRedirects;
      break; // The rest of this group would be on the wrong path.
    }
  }
  if (!Packet.Insts.empty() || Packet.Last) {
    ++NumGroups;
    OutInstsPort.send(Packet);
  }
  NextFetchCycle = Now + FetchInterval + (Redirect ? RedirectPenalty : 0);
  scheduleFetch();
}
