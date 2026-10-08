#pragma once

/** @file
 *  @brief Scalar execution: the integer and FP functional units, modelled as
 *         ResourcePool objects.
 *
 *  Handles every instruction Dispatch routes to ExecTarget::SCALAR: integer,
 *  branch, jump, CSR, fence, system, FP and unknown instructions. Memory
 *  instructions go to the LSU, vector instructions to the vector module.
 *
 *  Super-coarse behaviour, deliberately simpler than the M3 RTL:
 *  - one flat latency per class; no data-dependent divider latency;
 *  - pool contention is invisible to Dispatch: a busy unit only delays the
 *    start.
 *
 *  Result timing convention used by the scoreboard:
 *  result_ready_cycle = start + latency - 1, i.e. with latency 1 a dependent
 *  instruction can dispatch in the cycle the producer starts (back-to-back).
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

/** ScalarExec's parameters, copied out of the parameter set at construction.
 *
 *  Sparta requires every parameter to be read when the unit is built (so a
 *  misspelled config key fails loudly), so the unit reads them all into this
 *  struct up front. Each field has the meaning of the parameter of the same
 *  name in ScalarExec::ScalarExecParameterSet.
 */
struct ScalarExecConfig {
    std::uint32_t alu_count = 0;      ///< See ScalarExecParameterSet.
    std::uint32_t alu_latency = 0;    ///< See ScalarExecParameterSet.
    std::uint32_t mul_count = 0;      ///< See ScalarExecParameterSet.
    std::uint32_t mul_latency = 0;    ///< See ScalarExecParameterSet.
    std::uint32_t mul_occupancy = 0;  ///< See ScalarExecParameterSet.
    std::uint32_t div_latency = 0;    ///< See ScalarExecParameterSet.
    std::uint32_t div_occupancy = 0;  ///< See ScalarExecParameterSet.
    std::uint32_t fpu_latency = 0;    ///< See ScalarExecParameterSet.
    std::uint32_t fpu_occupancy = 0;  ///< See ScalarExecParameterSet.
    std::uint32_t fdiv_latency = 0;   ///< See ScalarExecParameterSet.
    std::uint32_t fdiv_occupancy = 0; ///< See ScalarExecParameterSet.
};

/// Scalar functional units. Tree location: top.backend.scalar_exec.
class ScalarExec : public sparta::Unit {
  public:
    /// Parameters of the scalar units (top.backend.scalar_exec.params).
    class ScalarExecParameterSet : public sparta::ParameterSet {
      public:
        /** Registers the parameters with the tree node.
         *
         *  @param node The parameter set's tree node.
         */
        explicit ScalarExecParameterSet(sparta::TreeNode* node)
            : sparta::ParameterSet(node) {}

        /// ALUs; they also run branches, jumps, CSRs and system ops.
        PARAMETER(std::uint32_t, alu_count, 4, "ALUs")
        /// ALU latency.
        PARAMETER(std::uint32_t, alu_latency, 1, "ALU latency")
        /// Multipliers.
        PARAMETER(std::uint32_t, mul_count, 1, "Multipliers")
        /// Multiply latency.
        PARAMETER(std::uint32_t, mul_latency, 2, "Multiply latency")
        /// Cycles a multiply blocks the multiplier.
        PARAMETER(std::uint32_t, mul_occupancy, 1,
                  "Cycles a multiply blocks the multiplier")
        /// Divide latency (flat; no early-out).
        PARAMETER(std::uint32_t, div_latency, 32, "Divide latency")
        /// Cycles a divide blocks the divider.
        PARAMETER(std::uint32_t, div_occupancy, 32,
                  "Cycles a divide blocks the divider")
        /// FP add/mul/fma/convert latency.
        PARAMETER(std::uint32_t, fpu_latency, 3, "FP op latency")
        /// Cycles an FP op blocks the FPU.
        PARAMETER(std::uint32_t, fpu_occupancy, 1,
                  "Cycles an FP op blocks the FPU")
        /// FP divide/sqrt latency.
        PARAMETER(std::uint32_t, fdiv_latency, 12, "FP divide/sqrt latency")
        /// Cycles an FP divide/sqrt blocks the divider.
        PARAMETER(std::uint32_t, fdiv_occupancy, 12,
                  "Cycles an FP divide/sqrt blocks the divider")
    };

    /// Name of this unit in the Sparta tree. Sparta's ResourceFactory
    /// requires a static member called exactly `name`.
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr const char* name = "scalar_exec";

    /** Creates the unit.
     *
     *  @param node Tree node the unit is attached to.
     *  @param params The unit's parameters.
     */
    ScalarExec(sparta::TreeNode* node, const ScalarExecParameterSet* params);

    /** All pools, for the end-of-run summary.
     *
     *  @return The pools in a fixed order: alu, mul, div, fpu, fdiv.
     */
    [[nodiscard]] std::vector<const ResourcePool*> pools() const;

  private:
    /** Port handler: an instruction was dispatched to the scalar units.
     *
     *  Books a pool, then writes the instruction's issue, result-ready and
     *  completion cycles into it.
     *
     *  @param inst The dispatched instruction.
     */
    void receive_inst(const InstPtr& inst);

    /** Reads every parameter into a ScalarExecConfig.
     *
     *  @param params The unit's parameter set.
     *  @return The parameter values.
     */
    static ScalarExecConfig read_config(const ScalarExecParameterSet* params);

    /// The unit's parameter values.
    const ScalarExecConfig cfg_;
    /// Integer ALUs.
    ResourcePool alu_;
    /// Multipliers.
    ResourcePool mul_;
    /// Integer divider.
    ResourcePool div_;
    /// FPU.
    ResourcePool fpu_;
    /// FP divide/sqrt unit.
    ResourcePool fdiv_;

    /// Dispatched instructions in from Dispatch.
    sparta::DataInPort<InstPtr> in_insts_{&unit_port_set_, "in_insts", 1};

    /// Statistic: instructions executed.
    sparta::Counter num_executed_{&unit_stat_set_, "num_executed",
                                  "Instructions executed",
                                  sparta::Counter::COUNT_NORMAL};
    /// Statistic: instructions of unknown class.
    sparta::Counter num_unknown_{&unit_stat_set_, "num_unknown_class",
                                 "Instructions of unknown class (timed as ALU)",
                                 sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf
