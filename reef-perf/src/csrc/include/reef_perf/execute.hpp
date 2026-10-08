#pragma once

/** @file
 *  @brief Execute unit: every execution resource in the core, modelled as
 *         ResourcePool objects.
 *
 *  Super-coarse behaviour, deliberately simpler than the M3 RTL:
 *  - one flat latency per class; no data-dependent divider latency;
 *  - the LSU is one pool: cost = distinct 16-byte lines touched x
 *    lsu_cycles_per_line, with no separate scalar and vector paths;
 *  - the vector unit is one pool of `vec_units` identical lanes. An
 *    instruction occupies a lane for (uops x vec_cycles_per_uop), where
 *    uops = ceil(vl x SEW / VLEN). No per-type functional units, no decode
 *    width and no vector ROB.
 *
 *  Scalar pool contention is invisible to Dispatch: a busy unit only delays
 *  the start. The LSU and vector pools have queues that Dispatch sees through
 *  credits.
 *
 *  Result timing convention used by the scoreboard:
 *  result_ready_cycle = start + latency - 1, i.e. with latency 1 a dependent
 *  instruction can dispatch in the cycle the producer starts (back-to-back).
 */

#include "reef_perf/inst.hpp"
#include "reef_perf/resource_pool.hpp"

#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

#include <cstdint>
#include <vector>

namespace reef_perf {

/** Execute's parameters, copied out of the parameter set at construction.
 *
 *  Sparta requires every parameter to be read when the unit is built (so a
 *  misspelled config key fails loudly), so Execute reads them all into this
 *  struct up front. Each field has the meaning of the parameter of the same
 *  name in Execute::ExecuteParameterSet.
 */
struct ExecuteConfig {
    std::uint32_t alu_count = 0;              ///< See ExecuteParameterSet.
    std::uint32_t alu_latency = 0;            ///< See ExecuteParameterSet.
    std::uint32_t mul_count = 0;              ///< See ExecuteParameterSet.
    std::uint32_t mul_latency = 0;            ///< See ExecuteParameterSet.
    std::uint32_t mul_occupancy = 0;          ///< See ExecuteParameterSet.
    std::uint32_t div_latency = 0;            ///< See ExecuteParameterSet.
    std::uint32_t div_occupancy = 0;          ///< See ExecuteParameterSet.
    std::uint32_t fpu_latency = 0;            ///< See ExecuteParameterSet.
    std::uint32_t fpu_occupancy = 0;          ///< See ExecuteParameterSet.
    std::uint32_t fdiv_latency = 0;           ///< See ExecuteParameterSet.
    std::uint32_t fdiv_occupancy = 0;         ///< See ExecuteParameterSet.
    std::uint32_t lsu_queue_entries = 0;      ///< See ExecuteParameterSet.
    std::uint32_t lsu_latency = 0;            ///< See ExecuteParameterSet.
    std::uint32_t lsu_line_bytes = 0;         ///< See ExecuteParameterSet.
    std::uint32_t lsu_cycles_per_line = 0;    ///< See ExecuteParameterSet.
    std::uint32_t vec_queue_entries = 0;      ///< See ExecuteParameterSet.
    std::uint32_t vec_units = 0;              ///< See ExecuteParameterSet.
    std::uint32_t vlen_bits = 0;              ///< See ExecuteParameterSet.
    std::uint32_t vec_latency = 0;            ///< See ExecuteParameterSet.
    std::uint32_t vec_cycles_per_uop = 0;     ///< See ExecuteParameterSet.
    std::uint32_t vec_div_cycles_per_uop = 0; ///< See ExecuteParameterSet.
    std::uint32_t vec_to_scalar_latency = 0;  ///< See ExecuteParameterSet.
};

/// Execute unit. Tree location: top.core.execute.
class Execute : public sparta::Unit {
  public:
    /// Parameters of the execute unit (top.core.execute.params).
    class ExecuteParameterSet : public sparta::ParameterSet {
      public:
        /** Registers the parameters with the tree node.
         *
         *  @param node The parameter set's tree node.
         */
        explicit ExecuteParameterSet(sparta::TreeNode* node)
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
        /// Vector command queue entries, as seen by Dispatch.
        PARAMETER(std::uint32_t, vec_queue_entries, 8,
                  "Vector command queue entries")
        /// Identical vector execution lanes.
        PARAMETER(std::uint32_t, vec_units, 2, "Vector execution lanes")
        /// Vector register length (VLEN) in bits.
        PARAMETER(std::uint32_t, vlen_bits, 128, "VLEN in bits")
        /// Latency of a one-uop vector op.
        PARAMETER(std::uint32_t, vec_latency, 4, "One-uop vector latency")
        /// Cycles a lane is busy per uop.
        PARAMETER(std::uint32_t, vec_cycles_per_uop, 1,
                  "Cycles a lane is busy per uop")
        /// Cycles per uop for vector divide and sqrt.
        PARAMETER(std::uint32_t, vec_div_cycles_per_uop, 8,
                  "Cycles per uop for vector divide/sqrt")
        /// Extra latency for vector results written to scalar registers.
        PARAMETER(std::uint32_t, vec_to_scalar_latency, 2,
                  "Extra latency for vector results written to x/f regs")
    };

    /// Name of this unit in the Sparta tree. Sparta's ResourceFactory
    /// requires a static member called exactly `name`.
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr const char* name = "execute";

    /** Creates the unit.
     *
     *  @param node Tree node the unit is attached to.
     *  @param params The unit's parameters.
     */
    Execute(sparta::TreeNode* node, const ExecuteParameterSet* params);

    /** All pools, for the end-of-run summary.
     *
     *  @return The pools in a fixed order: alu, mul, div, fpu, fdiv, lsu,
     *          vector.
     */
    [[nodiscard]] std::vector<const ResourcePool*> pools() const;

  private:
    /** Port handler: an instruction was dispatched to Execute.
     *
     *  Books a pool, then writes the instruction's issue, result-ready and
     *  completion cycles into it.
     *
     *  @param inst The dispatched instruction.
     */
    void receive_inst(const InstPtr& inst);

    /// Startup handler: tells Dispatch the LSU and vector queue sizes.
    void send_initial_credits();

    /** Reads every parameter into an ExecuteConfig.
     *
     *  @param params The unit's parameter set.
     *  @return The parameter values.
     */
    static ExecuteConfig read_config(const ExecuteParameterSet* params);

    /** Distinct memory lines an instruction touches.
     *
     *  @param inst A memory instruction.
     *  @return Number of lsu_line_bytes-sized lines (at least 1).
     */
    [[nodiscard]] std::uint32_t distinct_lines(const Inst& inst) const;

    /** Vector uops an instruction splits into.
     *
     *  @param inst A vector instruction.
     *  @return ceil(vl x SEW / VLEN), at least 1.
     */
    [[nodiscard]] std::uint32_t vector_uops(const Inst& inst) const;

    /// The unit's parameter values.
    const ExecuteConfig cfg_;
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
    /// Load/store unit.
    ResourcePool lsu_;
    /// Vector lanes.
    ResourcePool vec_;

    /// Dispatched instructions in from Dispatch.
    sparta::DataInPort<InstPtr> in_insts_{&unit_port_set_, "in_insts", 1};
    /// LSU queue credits out to Dispatch.
    sparta::DataOutPort<std::uint32_t> out_lsu_credits_{&unit_port_set_,
                                                        "out_lsu_credits"};
    /// Vector queue credits out to Dispatch.
    sparta::DataOutPort<std::uint32_t> out_vec_credits_{&unit_port_set_,
                                                        "out_vec_credits"};

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
