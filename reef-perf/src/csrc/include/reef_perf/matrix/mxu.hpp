#pragma once

/** @file
 *  @brief Matrix execution unit: a stub with the same interface as the
 *         vector unit.
 *
 *  Handles every instruction Dispatch routes to ExecTarget::MATRIX. No
 *  matrix instructions are encoded yet, so in today's workloads nothing
 *  reaches this unit; it exists so the interface between the backend and
 *  the matrix engine can be built and tested before the engine itself.
 *
 *  Stub behaviour: a command queue Dispatch sees through credits, and
 *  `mtx_units` engines that each take `mtx_occupancy` cycles per
 *  instruction with a flat `mtx_latency`. All values are placeholders.
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

/// Matrix execution unit. Tree location: top.matrix.mxu.
class Mxu : public sparta::Unit {
  public:
    /// Parameters of the matrix unit (top.matrix.mxu.params).
    class MxuParameterSet : public sparta::ParameterSet {
      public:
        /** Registers the parameters with the tree node.
         *
         *  @param node The parameter set's tree node.
         */
        explicit MxuParameterSet(sparta::TreeNode* node)
            : sparta::ParameterSet(node) {}

        /// Matrix command queue entries, as seen by Dispatch.
        PARAMETER(std::uint32_t, mtx_queue_entries, 4,
                  "Matrix command queue entries")
        /// Identical matrix engines.
        PARAMETER(std::uint32_t, mtx_units, 1, "Matrix engines")
        /// Latency of one matrix instruction.
        PARAMETER(std::uint32_t, mtx_latency, 8, "Matrix op latency")
        /// Cycles an instruction blocks its engine.
        PARAMETER(std::uint32_t, mtx_occupancy, 4,
                  "Cycles a matrix op blocks its engine")
    };

    /// Name of this unit in the Sparta tree. Sparta's ResourceFactory
    /// requires a static member called exactly `name`.
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr const char* name = "mxu";

    /** Creates the unit.
     *
     *  @param node Tree node the unit is attached to.
     *  @param params The unit's parameters.
     */
    Mxu(sparta::TreeNode* node, const MxuParameterSet* params);

    /** All pools, for the end-of-run summary.
     *
     *  @return {matrix}.
     */
    [[nodiscard]] std::vector<const ResourcePool*> pools() const {
        return {&engines_};
    }

  private:
    /** Port handler: a matrix instruction was dispatched to the unit.
     *
     *  @param inst The dispatched instruction.
     */
    void receive_inst(const InstPtr& inst);

    /// Startup handler: tells Dispatch how big the command queue is.
    void send_initial_credits();

    /// Value of the mtx_queue_entries parameter.
    const std::uint32_t queue_entries_;
    /// Value of the mtx_latency parameter.
    const std::uint32_t latency_;
    /// Value of the mtx_occupancy parameter.
    const std::uint32_t occupancy_;
    /// The matrix engines.
    ResourcePool engines_;

    /// Matrix instructions in from Dispatch.
    sparta::DataInPort<InstPtr> in_insts_{&unit_port_set_, "in_insts", 1};
    /// Command queue credits out to Dispatch.
    sparta::DataOutPort<std::uint32_t> out_credits_{&unit_port_set_,
                                                    "out_credits"};

    /// Statistic: matrix instructions executed.
    sparta::Counter num_executed_{&unit_stat_set_, "num_executed",
                                  "Matrix instructions executed",
                                  sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf
