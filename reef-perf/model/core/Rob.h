//===- Rob.h - The retirement buffer ----------------------------*- C++ -*-===//
///
/// \file
/// The retirement buffer. Instructions enter at dispatch and leave in program
/// order once complete; dispatch stops while it is full. Super-coarse: stores
/// retire when the LSU finishes them, there is no vector ROB, and CSRs do not
/// wait for an empty Rob.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_CORE_ROB_H
#define REEF_PERF_CORE_ROB_H

#include "common/Inst.h"

#include "sparta/events/UniqueEvent.hpp"
#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

#include <cstdint>
#include <deque>
#include <vector>

namespace reef_perf {

/// A retired instruction that read mcycle. The microbenchmarks read it at the
/// start and end of their region of interest, so differences between reads
/// are directly comparable with the RTL's.
struct McycleRead {
  Cycle IssueCycle; ///< The cycle it read mcycle.
  uint64_t Seq;     ///< Its dynamic instruction number.
};

/// The retirement buffer. Tree location: top.core.rob.
class Rob : public sparta::Unit {
public:
  // Parameter names are config keys, so they keep Sparta's snake_case.
  // NOLINTBEGIN(readability-identifier-naming)
  class RobParameterSet : public sparta::ParameterSet {
  public:
    explicit RobParameterSet(sparta::TreeNode *Node)
        : sparta::ParameterSet(Node) {}
    PARAMETER(uint32_t, rob_entries, 8, "Retirement buffer entries")
    PARAMETER(uint32_t, retire_width, 4, "Max instructions retired per cycle")
  };

  /// Sparta resource name; ResourceFactory requires this exact member name.
  static constexpr const char *name = "rob";
  // NOLINTEND(readability-identifier-naming)

  /// Creates the unit on \p Node with parameters \p Params.
  Rob(sparta::TreeNode *Node, const RobParameterSet *Params);

  /// \returns the instructions retired so far.
  uint64_t getNumRetired() const { return NumRetired.get(); }

  /// \returns the cycle of the last retirement: total cycles once the run
  /// ends.
  Cycle getLastRetireCycle() const { return LastRetireCycle; }

  /// \returns the oldest instruction not yet retired, or nullptr.
  InstRef getOldest() const {
    return Entries.empty() ? nullptr : Entries.front();
  }

  /// \returns every retired read of mcycle, in program order.
  const std::vector<McycleRead> &getMcycleReads() const { return McycleReads; }

private:
  void receiveInst(const InstRef &I);
  void retireInsts();
  void sendInitialCredits();

  const uint32_t RobEntries;
  const uint32_t RetireWidth;
  std::deque<InstRef> Entries; // Oldest first.
  Cycle LastRetireCycle = 0;
  std::vector<McycleRead> McycleReads;

  sparta::DataInPort<InstRef> InInstsPort{&unit_port_set_, "in_insts", 1};
  sparta::DataOutPort<uint32_t> OutCreditsPort{&unit_port_set_, "out_credits"};
  sparta::UniqueEvent<> RetireEvent{&unit_event_set_, "ev_retire",
                                    CREATE_SPARTA_HANDLER(Rob, retireInsts)};
  sparta::Counter NumRetired{&unit_stat_set_, "num_retired",
                             "Instructions retired",
                             sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf

#endif // REEF_PERF_CORE_ROB_H
