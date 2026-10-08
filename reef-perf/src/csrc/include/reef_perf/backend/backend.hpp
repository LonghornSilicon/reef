#pragma once

/** @file
 *  @brief Backend module: dispatch, scalar execution and retirement.
 *         Tree location: top.backend.
 *
 *  Units:
 *  - top.backend.dispatch  Dispatch (instruction buffer, scoreboard, credits)
 *  - top.backend.execute   Execute
 *  - top.backend.rob       Rob
 */

#include "reef_perf/backend/dispatch.hpp"
#include "reef_perf/common/module.hpp"
#include "reef_perf/common/resource_pool.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace reef_perf {

class Execute;
class Rob;

/// The backend module.
class Backend : public Module {
  public:
    /// Public port: fetch groups in from the frontend (FetchPacket).
    static constexpr const char* kInInsts = "dispatch.ports.in_insts";
    /// Public port: instruction-buffer credits out to the frontend.
    static constexpr const char* kOutFetchCredits =
        "dispatch.ports.out_fetch_credits";

    /// Creates the module; its tree node is top.backend.
    Backend();

    /** Registers the factories of the backend's units.
     *
     *  @param resources The simulation's resource set.
     */
    void add_factories(sparta::ResourceSet& resources) override;

    /// Binds Dispatch, Execute and the Rob together.
    void bind() override;

    /** The backend's pools.
     *
     *  @return Execute's pools.
     */
    [[nodiscard]] std::vector<const ResourcePool*> pools() const override;

    /** Instructions retired.
     *
     *  @return Instruction count.
     */
    [[nodiscard]] std::uint64_t num_retired() const;

    /** Cycle of the last retirement: total cycles once the run is over.
     *
     *  @return Cycle number.
     */
    [[nodiscard]] std::uint64_t last_retire_cycle() const;

    /** Cycles in which at least one instruction dispatched.
     *
     *  @return Cycle count.
     */
    [[nodiscard]] std::uint64_t cycles_with_dispatch() const;

    /** Cycles in which nothing dispatched, blamed on one reason.
     *
     *  @param reason The stall reason.
     *  @return Cycle count.
     */
    [[nodiscard]] std::uint64_t stall_cycles(StallReason reason) const;

  protected:
    /** The module's units.
     *
     *  @return {"dispatch", "execute", "rob"}.
     */
    [[nodiscard]] std::vector<std::string> unit_names() const override;

  private:
    /// Dispatch unit, valid after bind().
    Dispatch* dispatch_ = nullptr;
    /// Execute unit, valid after bind().
    Execute* execute_ = nullptr;
    /// Retirement buffer, valid after bind().
    Rob* rob_ = nullptr;
};

} // namespace reef_perf
