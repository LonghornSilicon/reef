#pragma once

/** @file
 *  @brief Vector execution unit: the command queue and the vector lanes.
 *
 *  Handles every instruction Dispatch routes to ExecTarget::VECTOR: vector
 *  arithmetic, permutes, vector-to-scalar moves and vset*. Vector loads and
 *  stores go to the backend's LSU, as on M3.
 *
 *  Super-coarse behaviour, deliberately simpler than the M3 RTL:
 *  - one pool of `vec_units` identical lanes. An instruction occupies a lane
 *    for (uops x vec_cycles_per_uop), where uops = ceil(vl x SEW / VLEN);
 *  - no per-type functional units, no decode width, no vector ROB and no
 *    chaining;
 *  - vset* takes `vset_latency` cycles and uses no lane;
 *  - Dispatch sees the command queue through credits; an entry frees when
 *    its instruction starts.
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

/** Vxu's parameters, copied out of the parameter set at construction.
 *
 *  Each field has the meaning of the parameter of the same name in
 *  Vxu::VxuParameterSet.
 */
struct VxuConfig {
    std::uint32_t vec_queue_entries = 0;      ///< See VxuParameterSet.
    std::uint32_t vec_units = 0;              ///< See VxuParameterSet.
    std::uint32_t vlen_bits = 0;              ///< See VxuParameterSet.
    std::uint32_t vec_latency = 0;            ///< See VxuParameterSet.
    std::uint32_t vec_cycles_per_uop = 0;     ///< See VxuParameterSet.
    std::uint32_t vec_div_cycles_per_uop = 0; ///< See VxuParameterSet.
    std::uint32_t vec_to_scalar_latency = 0;  ///< See VxuParameterSet.
    std::uint32_t vset_latency = 0;           ///< See VxuParameterSet.
};

/** Vector uops an instruction splits into.
 *
 *  @param inst A vector instruction.
 *  @param vlen_bits Vector register length in bits.
 *  @return ceil(vl x SEW / VLEN), at least 1.
 */
std::uint32_t vector_uops(const Inst& inst, std::uint32_t vlen_bits);

/// Vector execution unit. Tree location: top.vector.vxu.
class Vxu : public sparta::Unit {
  public:
    /// Parameters of the vector unit (top.vector.vxu.params).
    class VxuParameterSet : public sparta::ParameterSet {
      public:
        /** Registers the parameters with the tree node.
         *
         *  @param node The parameter set's tree node.
         */
        explicit VxuParameterSet(sparta::TreeNode* node)
            : sparta::ParameterSet(node) {}

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
        /// Latency of vsetvli, vsetivli and vsetvl.
        PARAMETER(std::uint32_t, vset_latency, 1, "vset* latency")
    };

    /// Name of this unit in the Sparta tree. Sparta's ResourceFactory
    /// requires a static member called exactly `name`.
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr const char* name = "vxu";

    /** Creates the unit.
     *
     *  @param node Tree node the unit is attached to.
     *  @param params The unit's parameters.
     */
    Vxu(sparta::TreeNode* node, const VxuParameterSet* params);

    /** All pools, for the end-of-run summary.
     *
     *  @return {vector}.
     */
    [[nodiscard]] std::vector<const ResourcePool*> pools() const {
        return {&lanes_};
    }

  private:
    /** Port handler: a vector instruction was dispatched to the unit.
     *
     *  @param inst The dispatched instruction.
     */
    void receive_inst(const InstPtr& inst);

    /// Startup handler: tells Dispatch how big the command queue is.
    void send_initial_credits();

    /** Reads every parameter into a VxuConfig.
     *
     *  @param params The unit's parameter set.
     *  @return The parameter values.
     */
    static VxuConfig read_config(const VxuParameterSet* params);

    /// The unit's parameter values.
    const VxuConfig cfg_;
    /// Vector lanes.
    ResourcePool lanes_;

    /// Vector instructions in from Dispatch.
    sparta::DataInPort<InstPtr> in_insts_{&unit_port_set_, "in_insts", 1};
    /// Command queue credits out to Dispatch.
    sparta::DataOutPort<std::uint32_t> out_credits_{&unit_port_set_,
                                                    "out_credits"};

    /// Statistic: vector instructions executed.
    sparta::Counter num_executed_{&unit_stat_set_, "num_executed",
                                  "Vector instructions executed",
                                  sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf
