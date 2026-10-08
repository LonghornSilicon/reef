#pragma once

/** @file
 *  @brief Rob: the retirement buffer.
 *
 *  Instructions enter at dispatch and leave in program order once they have
 *  completed. When the buffer is full, dispatch stops.
 *
 *  Super-coarse behaviour: stores retire when the LSU finishes them, there is
 *  no separate vector ROB, and CSRs do not wait for an empty Rob.
 */

#include "reef_perf/common/inst.hpp"

#include "sparta/events/UniqueEvent.hpp"
#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

#include <cstdint>
#include <deque>

namespace reef_perf {

/// Retirement buffer. Tree location: top.backend.rob.
class Rob : public sparta::Unit {
  public:
    /// Parameters of the retirement buffer (top.backend.rob.params).
    class RobParameterSet : public sparta::ParameterSet {
      public:
        /** Registers the parameters with the tree node.
         *
         *  @param node The parameter set's tree node.
         */
        explicit RobParameterSet(sparta::TreeNode* node)
            : sparta::ParameterSet(node) {}

        /// Retirement buffer entries.
        PARAMETER(std::uint32_t, rob_entries, 8, "Retirement buffer entries")
        /// Maximum instructions retired per cycle.
        PARAMETER(std::uint32_t, retire_width, 4,
                  "Max instructions retired per cycle")
    };

    /// Name of this unit in the Sparta tree. Sparta's ResourceFactory
    /// requires a static member called exactly `name`.
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr const char* name = "rob";

    /** Creates the unit.
     *
     *  @param node Tree node the unit is attached to.
     *  @param params The unit's parameters.
     */
    Rob(sparta::TreeNode* node, const RobParameterSet* params);

    /** Instructions retired so far.
     *
     *  @return Instruction count.
     */
    [[nodiscard]] std::uint64_t num_retired() const {
        return num_retired_.get();
    }

    /** Cycle of the most recent retirement, i.e. total program cycles once
     *  the simulation has finished.
     *
     *  @return Cycle number.
     */
    [[nodiscard]] std::uint64_t last_retire_cycle() const {
        return last_retire_cycle_;
    }

  private:
    /** Port handler: an instruction was dispatched and enters the buffer.
     *
     *  @param inst The dispatched instruction.
     */
    void receive_inst(const InstPtr& inst);

    /// Event handler: retires completed instructions from the head.
    void retire_insts();

    /// Startup handler: tells Dispatch how many entries are free.
    void send_initial_credits();

    /// Value of the rob_entries parameter.
    const std::uint32_t rob_entries_;
    /// Value of the retire_width parameter.
    const std::uint32_t retire_width_;
    /// In-flight instructions, oldest first.
    std::deque<InstPtr> rob_;
    /// Cycle of the most recent retirement.
    std::uint64_t last_retire_cycle_ = 0;

    /// Dispatched instructions in from Dispatch.
    sparta::DataInPort<InstPtr> in_insts_{&unit_port_set_, "in_insts", 1};
    /// Free-entry credits out to Dispatch.
    sparta::DataOutPort<std::uint32_t> out_credits_{&unit_port_set_,
                                                    "out_credits"};

    /// Event that runs retire_insts().
    sparta::UniqueEvent<> ev_retire_{&unit_event_set_, "ev_retire",
                                     CREATE_SPARTA_HANDLER(Rob, retire_insts)};

    /// Statistic: instructions retired.
    sparta::Counter num_retired_{&unit_stat_set_, "num_retired",
                                 "Instructions retired",
                                 sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf
