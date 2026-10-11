#pragma once

/** @file
 *  @brief Dispatch unit: holds the instruction buffer and sends instructions,
 *         in program order, to the unit that executes them (see
 *         exec_target()) and to the Rob.
 *
 *  Super-coarse behaviour, deliberately simpler than the M3 RTL:
 *  - up to `dispatch_width` instructions per cycle, strictly in order: the
 *    first instruction that cannot go blocks everything behind it;
 *  - an instruction waits if a register it reads or writes is still being
 *    produced (scoreboard: RAW and WAW hazards);
 *  - an instruction waits if the Rob, the LSU queue or the vector queue is
 *    full;
 *  - not modelled yet: Reef's special dispatch rules (a branch ends the group,
 *    FP and CSR instructions dispatch alone from slot 0, load/store address
 *    registers need a registered value, CSRs wait for an empty Rob).
 *
 *  Every cycle in which nothing dispatches is charged to exactly one stall
 *  reason, so the stall counters add up to (cycles - cycles_with_dispatch).
 */

#include "reef_perf/common/inst.hpp"
#include "reef_perf/common/interfaces.hpp"

#include "sparta/events/UniqueEvent.hpp"
#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

namespace reef_perf {

/// Why nothing dispatched in a cycle.
enum class StallReason : std::uint8_t {
    IBUF_EMPTY, ///< Nothing to dispatch: fetch is behind.
    RAW,        ///< A source register is not ready yet.
    WAW,        ///< A destination register still has a write in flight.
    ROB_FULL,   ///< No free Rob entry.
    LSU_FULL,   ///< LSU queue full.
    VEC_FULL,   ///< Vector command queue full.
    MTX_FULL,   ///< Matrix command queue full.
    NUM_REASONS ///< Number of reasons (not a reason).
};

/** Returns the name of a stall reason, as used in statistics and JSON.
 *
 *  @param reason The reason to name.
 *  @return A static string such as "raw_hazard".
 */
const char* stall_reason_name(StallReason reason);

/// Dispatch unit. Tree location: top.backend.dispatch.
class Dispatch : public sparta::Unit {
  public:
    /// Parameters of the dispatch unit (top.backend.dispatch.params).
    class DispatchParameterSet : public sparta::ParameterSet {
      public:
        /** Registers the parameters with the tree node.
         *
         *  @param node The parameter set's tree node.
         */
        explicit DispatchParameterSet(sparta::TreeNode* node)
            : sparta::ParameterSet(node) {}

        /// Maximum instructions dispatched per cycle.
        PARAMETER(std::uint32_t, dispatch_width, 4,
                  "Max instructions dispatched per cycle")
        /// Instruction buffer entries between fetch and dispatch.
        PARAMETER(std::uint32_t, ibuf_entries, 8,
                  "Instruction buffer entries between fetch and dispatch")
    };

    /// Name of this unit in the Sparta tree. Sparta's ResourceFactory
    /// requires a static member called exactly `name`.
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr const char* name = "dispatch";

    /** Creates the unit.
     *
     *  @param node Tree node the unit is attached to.
     *  @param params The unit's parameters.
     */
    Dispatch(sparta::TreeNode* node, const DispatchParameterSet* params);

    /** Instructions dispatched so far.
     *
     *  @return Instruction count.
     */
    [[nodiscard]] std::uint64_t num_dispatched() const {
        return num_dispatched_.get();
    }

    /** Cycles in which at least one instruction dispatched.
     *
     *  @return Cycle count.
     */
    [[nodiscard]] std::uint64_t cycles_with_dispatch() const {
        return cycles_with_dispatch_.get();
    }

    /** Cycles in which nothing dispatched, blamed on one reason.
     *
     *  @param reason The stall reason.
     *  @return Cycle count.
     */
    [[nodiscard]] std::uint64_t stall_cycles(StallReason reason) const {
        return stall_counters_.at(static_cast<std::size_t>(reason))->get();
    }

  private:
    /** Port handler: a fetch group arrived.
     *
     *  @param pkt The fetch group.
     */
    void receive_insts(const FetchPacket& pkt);

    /** Port handler: the Rob freed entries.
     *
     *  Sparta requires `const DataT&` handler parameters; see
     *  Fetch::receive_credits().
     *
     *  @param credits Number of entries freed.
     */
    void receive_rob_credits(const std::uint32_t& credits);

    /** Port handler: the LSU queue freed entries.
     *
     *  @param credits Number of entries freed.
     */
    void receive_lsu_credits(const std::uint32_t& credits);

    /** Port handler: the vector queue freed entries.
     *
     *  @param credits Number of entries freed.
     */
    void receive_vec_credits(const std::uint32_t& credits);

    /** Port handler: the matrix queue freed entries.
     *
     *  @param credits Number of entries freed.
     */
    void receive_mtx_credits(const std::uint32_t& credits);

    /// Startup handler: tells Fetch how big the instruction buffer is.
    void send_initial_credits();

    /// Event handler: dispatches up to dispatch_width instructions.
    void dispatch_group();

    /// Schedules dispatch_group() this cycle if there is work left.
    void schedule_dispatch();

    /** Checks whether an instruction can dispatch this cycle.
     *
     *  @param inst Instruction at the head of the buffer.
     *  @param now Current cycle.
     *  @param why Set to the blocking reason when the result is false.
     *  @return True if the instruction can dispatch.
     */
    bool can_dispatch(const InstPtr& inst, std::uint64_t now,
                      StallReason& why) const;

    /** Sends an instruction to the unit that executes it and spends that
     *  unit's queue credit, if it has a queue.
     *
     *  @param inst Instruction being dispatched.
     */
    void send_to_target(const InstPtr& inst);

    /// Value of the dispatch_width parameter.
    const std::uint32_t dispatch_width_;
    /// Value of the ibuf_entries parameter.
    const std::uint32_t ibuf_entries_;

    /// Instruction buffer, oldest first.
    std::deque<InstPtr> ibuf_;
    /// Whether Fetch has sent its last packet.
    bool fetch_done_ = false;

    /// Scoreboard: the in-flight instruction that last wrote each register.
    std::array<InstPtr, kNumRegs> last_writer_{};

    /// Free Rob entries.
    std::uint32_t rob_credits_ = 0;
    /// Free LSU queue entries.
    std::uint32_t lsu_credits_ = 0;
    /// Free vector queue entries.
    std::uint32_t vec_credits_ = 0;
    /// Free matrix queue entries.
    std::uint32_t mtx_credits_ = 0;

    /// Fetch groups in from Fetch.
    sparta::DataInPort<FetchPacket> in_insts_{&unit_port_set_, "in_insts", 1};
    /// Instruction-buffer credits out to Fetch.
    sparta::DataOutPort<std::uint32_t> out_fetch_credits_{&unit_port_set_,
                                                          "out_fetch_credits"};
    /// Dispatched instructions out to the scalar units (ExecTarget::SCALAR).
    sparta::DataOutPort<InstPtr> out_scalar_{&unit_port_set_, "out_scalar"};
    /// Dispatched instructions out to the LSU (ExecTarget::LSU).
    sparta::DataOutPort<InstPtr> out_lsu_{&unit_port_set_, "out_lsu"};
    /// Dispatched instructions out to the vector module
    /// (ExecTarget::VECTOR).
    sparta::DataOutPort<InstPtr> out_vector_{&unit_port_set_, "out_vector"};
    /// Dispatched instructions out to the matrix module
    /// (ExecTarget::MATRIX).
    sparta::DataOutPort<InstPtr> out_matrix_{&unit_port_set_, "out_matrix"};
    /// Dispatched instructions out to the Rob.
    sparta::DataOutPort<InstPtr> out_rob_{&unit_port_set_, "out_rob"};
    /// Rob credits in.
    sparta::DataInPort<std::uint32_t> in_rob_credits_{&unit_port_set_,
                                                      "in_rob_credits", 1};
    /// LSU queue credits in from the LSU.
    sparta::DataInPort<std::uint32_t> in_lsu_credits_{&unit_port_set_,
                                                      "in_lsu_credits", 1};
    /// Vector command-queue credits in from the vector module.
    sparta::DataInPort<std::uint32_t> in_vec_credits_{&unit_port_set_,
                                                      "in_vec_credits", 1};
    /// Matrix command-queue credits in from the matrix module.
    sparta::DataInPort<std::uint32_t> in_mtx_credits_{&unit_port_set_,
                                                      "in_mtx_credits", 1};

    /// Event that runs dispatch_group().
    sparta::UniqueEvent<> ev_dispatch_{
        &unit_event_set_, "ev_dispatch",
        CREATE_SPARTA_HANDLER(Dispatch, dispatch_group)};

    /// Statistic: instructions dispatched.
    sparta::Counter num_dispatched_{&unit_stat_set_, "num_dispatched",
                                    "Instructions dispatched",
                                    sparta::Counter::COUNT_NORMAL};
    /// Statistic: cycles with at least one dispatch.
    sparta::Counter cycles_with_dispatch_{
        &unit_stat_set_, "cycles_with_dispatch",
        "Cycles in which at least one instruction dispatched",
        sparta::Counter::COUNT_NORMAL};
    /// Statistics: stall cycles, one counter per StallReason.
    std::vector<std::unique_ptr<sparta::Counter>> stall_counters_;
};

} // namespace reef_perf
