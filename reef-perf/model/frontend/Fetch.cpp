//===- Fetch.cpp - The frontend module ------------------------------------===//

#include "frontend/Fetch.h"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <algorithm>
#include <cstdint>
#include <string>

using namespace reef_perf;

/// \returns true if M3's predecode (BTFN) predicts \p Encoding taken
/// (UncachedFetch.scala:87-100): jal, or a backward conditional branch with a
/// defined funct3. jalr is never predicted.
static bool isBtfnTaken(uint32_t Encoding) {
  uint32_t Opcode = Encoding & 0x7f;
  bool IsJal = Opcode == 0x6f;
  bool IsBackwardBranch = Opcode == 0x63 && ((Encoding >> 31) & 0x1) != 0 &&
                          ((Encoding >> 13) & 0x3) != 1;
  return IsJal || IsBackwardBranch;
}

/// \returns the predictor named \p Name: "none" or "btfn".
static Fetch::PredictorKind getPredictorKind(const std::string &Name) {
  if (Name == "btfn")
    return Fetch::PredictorKind::Btfn;
  sparta_assert(Name == "none", "frontend.predictor must be none or btfn");
  return Fetch::PredictorKind::None;
}

Fetch::Fetch(sparta::TreeNode *Node, const FetchParameterSet *Params)
    : sparta::Unit(Node, name), FetchWidth(Params->fetch_width),
      FetchInterval(Params->fetch_interval),
      ReorderBuffer(Params->fetch_reorder_buffer),
      MaxInFlight(Params->fetch_max_in_flight),
      ResponseCycles(Params->fetch_response_cycles),
      RedirectFromDispatch(Params->redirect_from_dispatch_cycles),
      RedirectPenalty(Params->redirect_penalty),
      Predictor(getPredictorKind(Params->predictor.getValue())),
      FetchBlockBytes(Params->fetch_block_bytes),
      IbufEntries(Params->ibuf_entries), Zvt(Params->zvt),
      UnaryVs1IsOpcode(Params->decode_unary_vs1_is_opcode) {
  sparta_assert(FetchWidth > 0 && FetchInterval > 0 && IbufEntries > 0);
  sparta_assert(!ReorderBuffer || (MaxInFlight > 0 && ResponseCycles > 0));
  sparta_assert(!ReorderBuffer || FetchWidth <= IbufEntries,
                "a fetched line must fit the instruction buffer");
  // The first group is fetched in cycle 1.
  sparta::StartupEvent(Node, CREATE_SPARTA_HANDLER(Fetch, scheduleFirstFetch));
}

void Fetch::scheduleFirstFetch() { FetchEvent.schedule(1 + StartCycle); }

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
  if (Ibuf.front() == WaitsForDispatch) {
    WaitsForDispatch = nullptr;
    NextIssue = Now + RedirectFromDispatch;
  }
  Ibuf.pop_front();
  if (FreedCycle != Now) {
    FreedCycle = Now;
    FreedAtCycle = 0;
  }
  ++FreedAtCycle;
  if (ReorderBuffer)
    return; // fetchSpeculative() runs every cycle and sees the dispatch.
  scheduleFetch(Now + 1); // The entry can be refilled next cycle.
}

bool Fetch::isDrained(Cycle Now) const {
  return Done && DoneCycle < Now && Ibuf.empty();
}

FrontendStall Fetch::getStall(Cycle /*Now*/) const {
  return LastGroupRedirected ? FrontendStall::BranchResteer
                             : FrontendStall::FetchBandwidth;
}

bool Fetch::isPredictedTaken(const Inst &I) const {
  switch (Predictor) {
  case PredictorKind::None:
    return false;
  case PredictorKind::Btfn:
    return isBtfnTaken(I.Encoding);
  }
  return false;
}

void Fetch::scheduleFetch(Cycle Earliest) {
  if (Done)
    return;
  Cycle Now = getClock()->currentCycle();
  FetchEvent.schedule(std::max(Earliest, NextFetchCycle) - Now);
}

void Fetch::fetchGroup() {
  sparta_assert(TheFuncSim, "FuncSim was not connected to Fetch");
  if (ReorderBuffer) {
    fetchSpeculative();
    return;
  }
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
  uint32_t Penalty = 0;
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
    bool Taken = I->isRedirect();
    bool PredictedTaken = isPredictedTaken(*I);
    bool BoundaryCross = FetchBlockBytes > 0 &&
                         I->NextPC / FetchBlockBytes != I->PC / FetchBlockBytes;
    if (Taken)
      ++NumRedirects;
    if (Taken != PredictedTaken) { // Wrong guess: the BRU redirects fetch.
      ++NumMispredicts;
      Penalty = RedirectPenalty;
    }
    if (Taken || PredictedTaken) {
      Redirect = true;
      break; // The rest of this group would be on the wrong path.
    }
    if (BoundaryCross)
      break; // The next instruction is in another block: the next fetch.
  }
  if (Fetched > 0) {
    ++NumGroups;
    LastGroupRedirected = Redirect;
    if (Mem) {
      const Address Line = Ibuf[Ibuf.size() - Fetched]->PC & ~Address{15};
      const MemAccess Access{Line, 16, false, 0};
      MemRequest Request;
      Request.Requester = RequesterKind::IFetch;
      Request.Accesses = std::span<const MemAccess>(&Access, 1);
      Request.Earliest = Now;
      Mem->access(Request);
    }
  } else if (Done) {
    ++NumGroups;
  }
  NextFetchCycle = Now + FetchInterval + Penalty;
  if (getFreeEntries(Now) > 0)
    scheduleFetch(NextFetchCycle);
}

void Fetch::fetchSpeculative() {
  Cycle Now = getClock()->currentCycle();
  if (Done)
    return;

  // The oldest line is written to the buffer when its response is back and
  // all of it fits (FetchControl: sufficientBuffer).
  if (!InFlight.empty() && InFlight.front().Ready <= Now) {
    Line &L = InFlight.front();
    if (getFreeEntries(Now) >= L.Insts.size()) {
      for (const InstRef &I : L.Insts) {
        I->FetchCycle = Now;
        Ibuf.push_back(I);
      }
      ++NumGroups;
      LastGroupRedirected = L.Redirects;
      // Predecode sees a predicted jump now and fetches its target in this
      // cycle; the sequential fetches behind it are dropped.
      bool WasPredictedJump = WaitsForCommit && L.Redirects;
      InFlight.pop_front();
      if (WasPredictedJump) {
        WaitsForCommit = false;
        std::erase_if(InFlight, [](const Line &X) { return X.WrongPath; });
        // A request held while the reorder buffer was full goes out now,
        // wasted, and the target is fetched the cycle after.
        if (PendingWrongPath)
          NextIssue = Now + 1;
        PendingWrongPath = false;
      }
    } else {
      ++CyclesNoCredit;
    }
  }

  // Behind a predicted jump whose line is still in flight, fetch runs on
  // down the next lines; they are dropped when the jump's line is written.
  if (WaitsForCommit && Now >= NextIssue) {
    if (InFlight.size() < MaxInFlight) {
      Line Wrong;
      Wrong.Ready = Now + ResponseCycles;
      Wrong.WrongPath = true;
      InFlight.push_back(std::move(Wrong));
    } else {
      PendingWrongPath = true;
    }
    NextIssue = Now + 1;
  }

  // Issue the next fetch. Behind a jump there is nothing useful to fetch
  // until its target is known: at its line's commit if predecode predicted
  // it, after the core dispatches it otherwise.
  if (!Exhausted && !WaitsForCommit && !WaitsForDispatch && Now >= NextIssue &&
      InFlight.size() < MaxInFlight) {
    Line L;
    L.Ready = Now + ResponseCycles;
    while (L.Insts.size() < FetchWidth) {
      InstRef I = TheFuncSim->next(); // Execute-at-fetch.
      if (!I) {
        Exhausted = true;
        break;
      }
      L.Insts.push_back(I);
      ++NumFetched;
      bool Taken = I->isRedirect();
      bool PredictedTaken = isPredictedTaken(*I);
      if (Taken)
        ++NumRedirects;
      if (Taken != PredictedTaken) {
        ++NumMispredicts;
        WaitsForDispatch = I;
      } else if (Taken) {
        WaitsForCommit = true;
      }
      if (Taken || PredictedTaken) {
        L.Redirects = true;
        break;
      }
      if (FetchBlockBytes > 0 &&
          I->NextPC / FetchBlockBytes != I->PC / FetchBlockBytes)
        break;
    }
    NextIssue = Now + 1;
    if (!L.Insts.empty()) {
      if (Mem) {
        const Address LineAddr = L.Insts.front()->PC & ~Address{15};
        const MemAccess Access{LineAddr, 16, false, 0};
        MemRequest Request;
        Request.Requester = RequesterKind::IFetch;
        Request.Accesses = std::span<const MemAccess>(&Access, 1);
        Request.Earliest = Now;
        Mem->access(Request);
      }
      InFlight.push_back(std::move(L));
    }
  }

  if (Exhausted && InFlight.empty()) {
    Done = true;
    DoneCycle = Now;
    ++NumGroups;
    return;
  }
  FetchEvent.schedule(1);
}
