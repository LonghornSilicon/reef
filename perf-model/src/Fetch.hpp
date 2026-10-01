// Fetch: pulls instructions from the functional simulator and sends them to
// Dispatch in groups.
//
// SUPER-COARSE BEHAVIOUR (deliberately simpler than the M3 RTL):
//   * Up to `fetch_width` instructions every `fetch_interval` cycles.
//   * A fetch group ends at any instruction that redirects the PC (taken
//     branch, jump, trap). The next group starts `redirect_penalty` cycles
//     later. There is no branch predictor: every redirect costs the same.
//   * Fetch stops when Dispatch's instruction buffer is full (credits).
//
// See docs/tickets-beginner.md for how M3 actually behaves.

#pragma once

#include "FuncSim.hpp"
#include "Inst.hpp"
#include "sparta/events/UniqueEvent.hpp"
#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

namespace coralnpu_perf {

class Fetch : public sparta::Unit {
 public:
  class FetchParameterSet : public sparta::ParameterSet {
   public:
    explicit FetchParameterSet(sparta::TreeNode* n) : sparta::ParameterSet(n) {}
    PARAMETER(uint32_t, fetch_width, 4, "Max instructions fetched per fetch cycle")
    PARAMETER(uint32_t, fetch_interval, 1, "Cycles between the starts of two fetch groups")
    PARAMETER(uint32_t, redirect_penalty, 1,
              "Extra cycles lost after any taken branch/jump before fetching again")
  };

  static const char* name;

  Fetch(sparta::TreeNode* node, const FetchParameterSet* p);

  // Called once by the simulation after the tree is built.
  void setFuncSim(FuncSim* funcsim) { funcsim_ = funcsim; }

  uint64_t numFetched() const { return num_fetched_.get(); }
  uint64_t numRedirects() const { return num_redirects_.get(); }

 private:
  void fetch_();
  void receiveCredits_(const uint32_t& credits);
  void scheduleFetch_();

  const uint32_t fetch_width_;
  const uint32_t fetch_interval_;
  const uint32_t redirect_penalty_;

  FuncSim* funcsim_ = nullptr;
  uint32_t credits_ = 0;       // free instruction-buffer entries in Dispatch
  uint64_t next_fetch_cycle_ = 0;
  bool done_ = false;

  sparta::DataOutPort<FetchPacket> out_insts_{&unit_port_set_, "out_insts"};
  sparta::DataInPort<uint32_t> in_credits_{&unit_port_set_, "in_credits", 1};

  sparta::UniqueEvent<> ev_fetch_{&unit_event_set_, "ev_fetch",
                                  CREATE_SPARTA_HANDLER(Fetch, fetch_)};

  sparta::Counter num_fetched_{&unit_stat_set_, "num_fetched",
                               "Instructions fetched", sparta::Counter::COUNT_NORMAL};
  sparta::Counter num_groups_{&unit_stat_set_, "num_groups",
                              "Fetch groups sent to dispatch", sparta::Counter::COUNT_NORMAL};
  sparta::Counter num_redirects_{&unit_stat_set_, "num_redirects",
                                 "Taken branches/jumps (each costs redirect_penalty)",
                                 sparta::Counter::COUNT_NORMAL};
  sparta::Counter cycles_no_credit_{&unit_stat_set_, "cycles_no_credit",
                                    "Fetch cycles lost because the instruction buffer was full",
                                    sparta::Counter::COUNT_NORMAL};
};

}  // namespace coralnpu_perf
