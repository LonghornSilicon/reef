//===- Fetch.h - The frontend module ----------------------------*- C++ -*-===//
///
/// \file
/// The frontend module: fetch. Every fetched instruction is executed in Spike
/// first (FuncSim), so the frontend knows each branch outcome.
///
/// Super-coarse, deliberately simpler than the M3 RTL:
/// - up to `fetch_width` instructions every `fetch_interval` cycles;
/// - a group ends at any instruction that redirects the PC (taken branch,
///   jump, trap), and the next group starts `redirect_penalty` cycles later;
///   there is no branch predictor, so every redirect costs the same;
/// - fetch stops while Dispatch's instruction buffer is full (credits).
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_FRONTEND_FETCH_H
#define REEF_PERF_FRONTEND_FETCH_H

#include "common/Inst.h"
#include "frontend/FuncSim.h"

#include "sparta/events/UniqueEvent.hpp"
#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

#include <cstdint>

namespace reef_perf {

/// The frontend module. Tree location: top.frontend.
class Fetch : public sparta::Unit {
public:
  // Parameter names are config keys, so they keep Sparta's snake_case.
  // NOLINTBEGIN(readability-identifier-naming)
  class FetchParameterSet : public sparta::ParameterSet {
  public:
    explicit FetchParameterSet(sparta::TreeNode *Node)
        : sparta::ParameterSet(Node) {}
    PARAMETER(uint32_t, fetch_width, 4,
              "Max instructions fetched per fetch cycle")
    PARAMETER(uint32_t, fetch_interval, 1,
              "Cycles between the starts of two fetch groups")
    PARAMETER(uint32_t, redirect_penalty, 1,
              "Extra cycles lost after any taken branch/jump before fetching "
              "again")
  };

  /// Sparta resource name; ResourceFactory requires this exact member name.
  static constexpr const char *name = "fetch";
  // NOLINTEND(readability-identifier-naming)

  /// Public port: fetch groups out to the backend (FetchPacket).
  static constexpr const char *OutInsts = "ports.out_insts";
  /// Public port: instruction-buffer credits in from the backend.
  static constexpr const char *InCredits = "ports.in_credits";

  /// Creates the unit on \p Node with parameters \p Params.
  Fetch(sparta::TreeNode *Node, const FetchParameterSet *Params);

  /// Connects \p Sim, which must outlive the unit.
  void setFuncSim(FuncSim *Sim) { TheFuncSim = Sim; }

  /// \returns the taken branches, jumps and traps fetched so far.
  uint64_t getNumRedirects() const { return NumRedirects.get(); }

private:
  void fetchGroup();
  // Sparta handlers take `const DataT &`.
  void receiveCredits(const uint32_t &NewCredits);
  void scheduleFetch();

  const uint32_t FetchWidth;
  const uint32_t FetchInterval;
  const uint32_t RedirectPenalty;

  FuncSim *TheFuncSim = nullptr;
  uint32_t Credits = 0;        // Free instruction-buffer entries.
  uint64_t NextFetchCycle = 0; // Earliest start of the next group.
  bool Done = false;           // The program is fully fetched.

  sparta::DataOutPort<FetchPacket> OutInstsPort{&unit_port_set_, "out_insts"};
  sparta::DataInPort<uint32_t> InCreditsPort{&unit_port_set_, "in_credits", 1};
  sparta::UniqueEvent<> FetchEvent{&unit_event_set_, "ev_fetch",
                                   CREATE_SPARTA_HANDLER(Fetch, fetchGroup)};

  sparta::Counter NumFetched{&unit_stat_set_, "num_fetched",
                             "Instructions fetched",
                             sparta::Counter::COUNT_NORMAL};
  sparta::Counter NumGroups{&unit_stat_set_, "num_groups",
                            "Fetch groups sent to dispatch",
                            sparta::Counter::COUNT_NORMAL};
  sparta::Counter NumRedirects{
      &unit_stat_set_, "num_redirects",
      "Taken branches/jumps (each costs redirect_penalty)",
      sparta::Counter::COUNT_NORMAL};
  sparta::Counter CyclesNoCredit{
      &unit_stat_set_, "cycles_no_credit",
      "Fetch cycles lost because the instruction buffer was full",
      sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf

#endif // REEF_PERF_FRONTEND_FETCH_H
