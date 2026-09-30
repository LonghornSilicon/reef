// Dispatch: holds the instruction buffer and sends instructions, in program
// order, to Execute and the reorder buffer (Rob).
//
// SUPER-COARSE BEHAVIOUR (deliberately simpler than the M3 RTL):
//   * Up to `dispatch_width` instructions per cycle, strictly in order: the
//     first instruction that cannot go blocks everything behind it.
//   * An instruction waits if a register it reads or writes is still being
//     produced (scoreboard: RAW and WAW hazards).
//   * An instruction waits if the Rob, the LSU queue or the vector queue is
//     full.
//   * NOT modelled yet: CoralNPU's special dispatch rules (branch ends the
//     group, FP/CSR dispatch alone from slot 0, load/store address registers
//     need a registered value, CSRs wait for an empty Rob). See the tickets.
//
// Every cycle in which *nothing* dispatches is charged to exactly one stall
// reason, so the stall counters add up to (cycles - cycles_with_dispatch).

#pragma once

#include <array>
#include <deque>
#include <memory>
#include <vector>

#include "Inst.hpp"
#include "sparta/events/UniqueEvent.hpp"
#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

namespace coralnpu_perf {

enum class StallReason : uint8_t {
  IBUF_EMPTY,   // nothing to dispatch: fetch is behind (redirects, fetch rate)
  RAW,          // a source register is not ready yet
  WAW,          // a destination register still has a write in flight
  ROB_FULL,     // no free reorder-buffer entry
  LSU_FULL,     // LSU queue full
  VEC_FULL,     // vector command queue full
  NUM_REASONS
};
const char* stallReasonName(StallReason r);

class Dispatch : public sparta::Unit {
 public:
  class DispatchParameterSet : public sparta::ParameterSet {
   public:
    explicit DispatchParameterSet(sparta::TreeNode* n) : sparta::ParameterSet(n) {}
    PARAMETER(uint32_t, dispatch_width, 4, "Max instructions dispatched per cycle")
    PARAMETER(uint32_t, ibuf_entries, 8, "Instruction buffer entries between fetch and dispatch")
  };

  static const char* name;

  Dispatch(sparta::TreeNode* node, const DispatchParameterSet* p);

  uint64_t numDispatched() const { return num_dispatched_.get(); }
  uint64_t cyclesWithDispatch() const { return cycles_with_dispatch_.get(); }
  uint64_t stallCycles(StallReason r) const {
    return stall_counters_[static_cast<size_t>(r)]->get();
  }

 private:
  void receiveInsts_(const FetchPacket& pkt);
  void receiveRobCredits_(const uint32_t& n);
  void receiveLsuCredits_(const uint32_t& n);
  void receiveVecCredits_(const uint32_t& n);
  void sendInitialCredits_();
  void dispatch_();
  void scheduleDispatch_();

  // Returns true if `inst` can dispatch this cycle; otherwise sets `why`.
  bool canDispatch_(const InstPtr& inst, uint64_t now, StallReason& why) const;

  const uint32_t dispatch_width_;
  const uint32_t ibuf_entries_;

  std::deque<InstPtr> ibuf_;
  bool fetch_done_ = false;

  // Scoreboard: the in-flight instruction that last wrote each register.
  std::array<InstPtr, kNumRegs> last_writer_{};

  uint32_t rob_credits_ = 0;
  uint32_t lsu_credits_ = 0;
  uint32_t vec_credits_ = 0;

  sparta::DataInPort<FetchPacket> in_insts_{&unit_port_set_, "in_insts", 1};
  sparta::DataOutPort<uint32_t> out_fetch_credits_{&unit_port_set_, "out_fetch_credits"};
  sparta::DataOutPort<InstPtr> out_execute_{&unit_port_set_, "out_execute"};
  sparta::DataOutPort<InstPtr> out_rob_{&unit_port_set_, "out_rob"};
  sparta::DataInPort<uint32_t> in_rob_credits_{&unit_port_set_, "in_rob_credits", 1};
  sparta::DataInPort<uint32_t> in_lsu_credits_{&unit_port_set_, "in_lsu_credits", 1};
  sparta::DataInPort<uint32_t> in_vec_credits_{&unit_port_set_, "in_vec_credits", 1};

  sparta::UniqueEvent<> ev_dispatch_{&unit_event_set_, "ev_dispatch",
                                     CREATE_SPARTA_HANDLER(Dispatch, dispatch_)};

  sparta::Counter num_dispatched_{&unit_stat_set_, "num_dispatched",
                                  "Instructions dispatched", sparta::Counter::COUNT_NORMAL};
  sparta::Counter cycles_with_dispatch_{&unit_stat_set_, "cycles_with_dispatch",
                                        "Cycles in which at least one instruction dispatched",
                                        sparta::Counter::COUNT_NORMAL};
  std::vector<std::unique_ptr<sparta::Counter>> stall_counters_;
};

}  // namespace coralnpu_perf
