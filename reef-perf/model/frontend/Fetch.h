//===- Fetch.h - The frontend module ----------------------------*- C++ -*-===//
///
/// \file
/// The frontend module: fetch and the instruction buffer. Every fetched
/// instruction is executed in Spike first (FuncSim), so the frontend knows
/// each branch outcome. The core takes instructions from the buffer
/// through InstBufferInterface; on M3 the buffer is part of fetch
/// (UncachedFetch.scala, InstructionBuffer).
///
/// Super-coarse, deliberately simpler than the M3 RTL:
/// - up to `fetch_width` instructions every `fetch_interval` cycles, while
///   the `ibuf_entries`-entry buffer has room;
/// - a group ends at any instruction that redirects the PC (taken branch,
///   jump, trap) or that the predictor predicts taken;
/// - with `predictor: none` every redirect costs `redirect_penalty` extra
///   cycles. With `predictor: static_btfn` (M3 predecode,
///   UncachedFetch.scala:87-100) a correctly predicted taken jump costs
///   `taken_bubble` and a mispredict costs `mispredict_penalty`.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_FRONTEND_FETCH_H
#define REEF_PERF_FRONTEND_FETCH_H

#include "common/Inst.h"
#include "common/Interfaces.h"
#include "frontend/FuncSim.h"

#include "sparta/events/UniqueEvent.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

#include <cstdint>
#include <deque>
#include <optional>
#include <string>

namespace reef_perf {

/// The frontend module. Tree location: top.frontend.
class Fetch : public sparta::Unit, public InstBufferInterface {
public:
  // Parameter names are config keys, so they keep Sparta's snake_case.
  // NOLINTBEGIN(readability-identifier-naming)
  struct FetchParameterSet : public sparta::ParameterSet {
    explicit FetchParameterSet(sparta::TreeNode *Node)
        : sparta::ParameterSet(Node) {}
    PARAMETER(uint32_t, fetch_width, 4,
              "Max instructions fetched per fetch cycle")
    PARAMETER(uint32_t, fetch_interval, 1,
              "Cycles between the starts of two fetch groups")
    PARAMETER(uint32_t, redirect_penalty, 1,
              "Extra cycles lost after any taken branch/jump before fetching "
              "again (predictor none)")
    PARAMETER(std::string, predictor, "none",
              "Branch predictor: none (every redirect costs redirect_penalty) "
              "or static_btfn (jal and backward branches predicted taken)")
    PARAMETER(uint32_t, taken_bubble, 0,
              "Extra cycles after a correctly predicted taken branch/jump "
              "(static_btfn)")
    PARAMETER(uint32_t, mispredict_penalty, 1,
              "Extra cycles after a mispredicted branch/jump or a trap "
              "(static_btfn)")
    PARAMETER(uint32_t, ibuf_entries, 8, "Instruction buffer entries")
  };

  /// Sparta resource name; ResourceFactory requires this exact member name.
  static constexpr const char *name = "fetch";
  // NOLINTEND(readability-identifier-naming)

  /// Creates the unit on \p Node with parameters \p Params.
  Fetch(sparta::TreeNode *Node, const FetchParameterSet *Params);

  /// Connects \p Sim, which must outlive the unit.
  void setFuncSim(FuncSim *Sim) { TheFuncSim = Sim; }

  /// \returns the taken branches, jumps and traps fetched so far.
  uint64_t getNumRedirects() const { return NumRedirects.get(); }

  /// \returns the oldest buffered instruction fetched before \p Now.
  InstRef peek(Cycle Now) const override;
  /// Removes the oldest instruction; its entry refills from \p Now + 1.
  void pop(Cycle Now) override;
  /// \returns true once fetch finished before \p Now and the buffer is empty.
  bool isDrained(Cycle Now) const override;
  /// \returns BranchResteer if the latest fetch group ended in a redirect or
  /// a mispredict, and FetchBandwidth otherwise.
  FrontendStall getStall(Cycle Now) const override;

private:
  void fetchGroup();
  void scheduleFirstFetch();
  void scheduleFetch(Cycle Earliest);
  /// \returns the entries fetch may fill at \p Now. Entries the core freed
  /// at \p Now only count from the next cycle.
  uint32_t getFreeEntries(Cycle Now) const;
  /// \returns the extra cycles before the next group if \p I ends the
  /// current one, or nullopt if the group continues. Updates the counters.
  std::optional<uint32_t> getGroupEndPenalty(const Inst &I);

  const uint32_t FetchWidth;
  const uint32_t FetchInterval;
  const uint32_t RedirectPenalty;
  const bool StaticBtfn;
  const uint32_t TakenBubble;
  const uint32_t MispredictPenalty;
  const uint32_t IbufEntries;

  FuncSim *TheFuncSim = nullptr;
  std::deque<InstRef> Ibuf;  // Oldest first.
  Cycle FreedCycle = 0;      // Cycle of the latest pop()s ...
  uint32_t FreedAtCycle = 0; // ... and how many entries they freed.
  Cycle NextFetchCycle = 0;  // Earliest start of the next group.
  bool LastGroupRedirected = false;
  bool Done = false;   // The program is fully fetched ...
  Cycle DoneCycle = 0; // ... as of this cycle.

  sparta::UniqueEvent<> FetchEvent{&unit_event_set_, "ev_fetch",
                                   CREATE_SPARTA_HANDLER(Fetch, fetchGroup)};

  sparta::Counter NumFetched{&unit_stat_set_, "num_fetched",
                             "Instructions fetched",
                             sparta::Counter::COUNT_NORMAL};
  sparta::Counter NumGroups{&unit_stat_set_, "num_groups",
                            "Fetch groups added to the instruction buffer",
                            sparta::Counter::COUNT_NORMAL};
  sparta::Counter NumRedirects{&unit_stat_set_, "num_redirects",
                               "Taken branches, jumps and traps",
                               sparta::Counter::COUNT_NORMAL};
  sparta::Counter NumPredictedTaken{
      &unit_stat_set_, "num_predicted_taken",
      "Branches/jumps the static predictor predicted taken",
      sparta::Counter::COUNT_NORMAL};
  sparta::Counter NumMispredicts{
      &unit_stat_set_, "num_mispredicts",
      "Mispredicted branches/jumps and traps (each costs mispredict_penalty)",
      sparta::Counter::COUNT_NORMAL};
  sparta::Counter NumJalr{&unit_stat_set_, "num_jalr",
                          "jalr instructions (never predicted)",
                          sparta::Counter::COUNT_NORMAL};
  sparta::Counter CyclesNoCredit{
      &unit_stat_set_, "cycles_no_credit",
      "Fetch cycles lost because the instruction buffer was full",
      sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf

#endif // REEF_PERF_FRONTEND_FETCH_H
