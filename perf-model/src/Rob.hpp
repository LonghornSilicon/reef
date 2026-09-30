// Rob: the retirement buffer. Instructions enter at dispatch and leave in
// program order once they have completed. When it is full, dispatch stops.
//
// SUPER-COARSE BEHAVIOUR: stores retire when the LSU finishes them; there is
// no separate vector ROB, and CSRs do not wait for an empty Rob.

#pragma once

#include <deque>

#include "Inst.hpp"
#include "sparta/events/StartupEvent.hpp"
#include "sparta/events/UniqueEvent.hpp"
#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

namespace coralnpu_perf {

class Rob : public sparta::Unit {
 public:
  class RobParameterSet : public sparta::ParameterSet {
   public:
    explicit RobParameterSet(sparta::TreeNode* n) : sparta::ParameterSet(n) {}
    PARAMETER(uint32_t, rob_entries, 8, "Retirement buffer entries")
    PARAMETER(uint32_t, retire_width, 4, "Max instructions retired per cycle")
  };

  static const char* name;

  Rob(sparta::TreeNode* node, const RobParameterSet* p);

  uint64_t numRetired() const { return num_retired_.get(); }
  // Cycle of the last retirement = total program cycles.
  uint64_t lastRetireCycle() const { return last_retire_cycle_; }

 private:
  void receiveInst_(const InstPtr& inst);
  void retire_();
  void sendInitialCredits_();

  const uint32_t rob_entries_;
  const uint32_t retire_width_;
  std::deque<InstPtr> rob_;
  uint64_t last_retire_cycle_ = 0;

  sparta::DataInPort<InstPtr> in_insts_{&unit_port_set_, "in_insts", 1};
  sparta::DataOutPort<uint32_t> out_credits_{&unit_port_set_, "out_credits"};

  sparta::UniqueEvent<> ev_retire_{&unit_event_set_, "ev_retire",
                                   CREATE_SPARTA_HANDLER(Rob, retire_)};

  sparta::Counter num_retired_{&unit_stat_set_, "num_retired",
                               "Instructions retired", sparta::Counter::COUNT_NORMAL};
};

}  // namespace coralnpu_perf
