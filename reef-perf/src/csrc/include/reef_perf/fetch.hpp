#pragma once

/** @file
 *  @brief Fetch unit: pulls instructions from the functional simulator and
 *         sends them to Dispatch in groups.
 *
 *  Super-coarse behaviour, deliberately simpler than the M3 RTL:
 *  - up to `fetch_width` instructions every `fetch_interval` cycles;
 *  - a group ends at any instruction that redirects the PC (taken branch,
 *    jump, trap), and the next group starts `redirect_penalty` cycles later.
 *    There is no branch predictor, so every redirect costs the same;
 *  - fetch stops when Dispatch's instruction buffer is full (credits).
 *
 *  See docs/tickets-beginner.md for how M3 actually behaves.
 */

#include "reef_perf/func_sim.hpp"
#include "reef_perf/inst.hpp"

#include "sparta/events/UniqueEvent.hpp"
#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

#include <cstdint>

namespace reef_perf {

/// Fetch unit. Tree location: top.core.fetch.
class Fetch : public sparta::Unit {
  public:
    /// Parameters of the fetch unit (top.core.fetch.params).
    class FetchParameterSet : public sparta::ParameterSet {
      public:
        /** Registers the parameters with the tree node.
         *
         *  @param node The parameter set's tree node.
         */
        explicit FetchParameterSet(sparta::TreeNode* node)
            : sparta::ParameterSet(node) {}

        /// Maximum instructions fetched per fetch cycle.
        PARAMETER(std::uint32_t, fetch_width, 4,
                  "Max instructions fetched per fetch cycle")
        /// Cycles between the starts of two fetch groups.
        PARAMETER(std::uint32_t, fetch_interval, 1,
                  "Cycles between the starts of two fetch groups")
        /// Extra cycles lost after a taken branch or jump.
        PARAMETER(std::uint32_t, redirect_penalty, 1,
                  "Extra cycles lost after any taken branch/jump before "
                  "fetching again")
    };

    /// Name of this unit in the Sparta tree. Sparta's ResourceFactory
    /// requires a static member called exactly `name`.
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr const char* name = "fetch";

    /** Creates the unit.
     *
     *  @param node Tree node the unit is attached to.
     *  @param params The unit's parameters.
     */
    Fetch(sparta::TreeNode* node, const FetchParameterSet* params);

    /** Connects the functional simulator. Called once, after the tree is
     *  built and before the simulation runs.
     *
     *  @param funcsim Functional simulator; must outlive this unit.
     */
    void set_func_sim(FuncSim* funcsim) { funcsim_ = funcsim; }

    /** Instructions fetched so far.
     *
     *  @return Instruction count.
     */
    [[nodiscard]] std::uint64_t num_fetched() const {
        return num_fetched_.get();
    }

    /** Taken branches, jumps and traps seen so far.
     *
     *  @return Redirect count.
     */
    [[nodiscard]] std::uint64_t num_redirects() const {
        return num_redirects_.get();
    }

  private:
    /// Event handler: fetches one group, if the unit is allowed to.
    void fetch_group();

    /** Port handler: Dispatch freed instruction-buffer entries.
     *
     *  The parameter is `const std::uint32_t&` rather than `std::uint32_t`
     *  because Sparta requires it: DataInPort handlers are built with
     *  CREATE_SPARTA_HANDLER_WITH_DATA, which only accepts member functions
     *  of type `void (T::*)(const DataT&)` (sparta/kernel/SpartaHandler.hpp).
     *
     *  @param credits Number of entries freed.
     */
    void receive_credits(const std::uint32_t& credits);

    /// Schedules fetch_group() for the next cycle fetch is allowed to run.
    void schedule_fetch();

    /// Value of the fetch_width parameter.
    const std::uint32_t fetch_width_;
    /// Value of the fetch_interval parameter.
    const std::uint32_t fetch_interval_;
    /// Value of the redirect_penalty parameter.
    const std::uint32_t redirect_penalty_;

    /// Functional simulator; set by set_func_sim().
    FuncSim* funcsim_ = nullptr;
    /// Free instruction-buffer entries in Dispatch.
    std::uint32_t credits_ = 0;
    /// Earliest cycle the next fetch group may start.
    std::uint64_t next_fetch_cycle_ = 0;
    /// Whether the program has been fully fetched.
    bool done_ = false;

    /// Fetch groups out to Dispatch.
    sparta::DataOutPort<FetchPacket> out_insts_{&unit_port_set_, "out_insts"};
    /// Instruction-buffer credits in from Dispatch.
    sparta::DataInPort<std::uint32_t> in_credits_{&unit_port_set_, "in_credits",
                                                  1};

    /// Event that runs fetch_group().
    sparta::UniqueEvent<> ev_fetch_{&unit_event_set_, "ev_fetch",
                                    CREATE_SPARTA_HANDLER(Fetch, fetch_group)};

    /// Statistic: instructions fetched.
    sparta::Counter num_fetched_{&unit_stat_set_, "num_fetched",
                                 "Instructions fetched",
                                 sparta::Counter::COUNT_NORMAL};
    /// Statistic: fetch groups sent.
    sparta::Counter num_groups_{&unit_stat_set_, "num_groups",
                                "Fetch groups sent to dispatch",
                                sparta::Counter::COUNT_NORMAL};
    /// Statistic: taken branches, jumps and traps.
    sparta::Counter num_redirects_{
        &unit_stat_set_, "num_redirects",
        "Taken branches/jumps (each costs redirect_penalty)",
        sparta::Counter::COUNT_NORMAL};
    /// Statistic: fetch attempts blocked by a full instruction buffer.
    sparta::Counter cycles_no_credit_{
        &unit_stat_set_, "cycles_no_credit",
        "Fetch cycles lost because the instruction buffer was full",
        sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf
