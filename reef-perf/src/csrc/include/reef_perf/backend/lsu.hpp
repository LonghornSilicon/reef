#pragma once

/** @file
 *  @brief Load/store unit: the queue and the single slot every scalar, FP
 *         and vector memory instruction goes through.
 *
 *  Super-coarse behaviour, deliberately simpler than the M3 RTL:
 *  - one slot: an instruction holds it for (distinct lines touched x
 *    lsu_cycles_per_line) cycles, using the addresses Spike recorded;
 *  - load-to-use latency is lsu_latency + occupancy - 1;
 *  - Dispatch sees the queue through credits; an entry frees when its
 *    instruction starts.
 */

#include "reef_perf/common/inst.hpp"
#include "reef_perf/common/resource_pool.hpp"

#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

#include <cstdint>
#include <vector>

namespace reef_perf {

/// Load/store unit. Tree location: top.backend.lsu.
class Lsu : public sparta::Unit {
  public:
    /// Parameters of the LSU (top.backend.lsu.params).
    class LsuParameterSet : public sparta::ParameterSet {
      public:
        /** Registers the parameters with the tree node.
         *
         *  @param node The parameter set's tree node.
         */
        explicit LsuParameterSet(sparta::TreeNode* node)
            : sparta::ParameterSet(node) {}

        /// LSU queue entries, as seen by Dispatch.
        PARAMETER(std::uint32_t, lsu_queue_entries, 4, "LSU queue entries")
        /// Load-to-use latency of a single-line access.
        PARAMETER(std::uint32_t, lsu_latency, 2, "Load-to-use latency")
        /// Bytes per memory transaction.
        PARAMETER(std::uint32_t, lsu_line_bytes, 16,
                  "Bytes per memory transaction")
        /// Cycles the LSU is busy per line transaction.
        PARAMETER(std::uint32_t, lsu_cycles_per_line, 1,
                  "Cycles the LSU is busy per line transaction")
    };

    /// Name of this unit in the Sparta tree. Sparta's ResourceFactory
    /// requires a static member called exactly `name`.
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr const char* name = "lsu";

    /** Creates the unit.
     *
     *  @param node Tree node the unit is attached to.
     *  @param params The unit's parameters.
     */
    Lsu(sparta::TreeNode* node, const LsuParameterSet* params);

    /** All pools, for the end-of-run summary.
     *
     *  @return {lsu}.
     */
    [[nodiscard]] std::vector<const ResourcePool*> pools() const {
        return {&slot_};
    }

  private:
    /** Port handler: a memory instruction was dispatched to the LSU.
     *
     *  @param inst The dispatched instruction.
     */
    void receive_inst(const InstPtr& inst);

    /// Startup handler: tells Dispatch how big the queue is.
    void send_initial_credits();

    /** Distinct memory lines an instruction touches.
     *
     *  @param inst A memory instruction.
     *  @return Number of lsu_line_bytes-sized lines (at least 1).
     */
    [[nodiscard]] std::uint32_t distinct_lines(const Inst& inst) const;

    /// Value of the lsu_queue_entries parameter.
    const std::uint32_t queue_entries_;
    /// Value of the lsu_latency parameter.
    const std::uint32_t latency_;
    /// Value of the lsu_line_bytes parameter.
    const std::uint32_t line_bytes_;
    /// Value of the lsu_cycles_per_line parameter.
    const std::uint32_t cycles_per_line_;
    /// The LSU slot.
    ResourcePool slot_;

    /// Memory instructions in from Dispatch.
    sparta::DataInPort<InstPtr> in_insts_{&unit_port_set_, "in_insts", 1};
    /// Queue credits out to Dispatch.
    sparta::DataOutPort<std::uint32_t> out_credits_{&unit_port_set_,
                                                    "out_credits"};

    /// Statistic: memory instructions executed.
    sparta::Counter num_executed_{&unit_stat_set_, "num_executed",
                                  "Memory instructions executed",
                                  sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf
