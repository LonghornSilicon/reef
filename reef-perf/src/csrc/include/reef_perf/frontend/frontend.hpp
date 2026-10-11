#pragma once

/** @file
 *  @brief Frontend module: fetch, the functional simulator (Spike) and
 *         decode. Tree location: top.frontend.
 *
 *  Units: top.frontend.fetch (Fetch).
 *
 *  The frontend creates every Inst (execute-at-fetch) and fills in the fields
 *  listed as frontend-owned in docs/interfaces.md. Spike and decode are plain
 *  C++ objects, not Sparta units.
 */

#include "reef_perf/common/module.hpp"
#include "reef_perf/frontend/fetch.hpp"
#include "reef_perf/frontend/func_sim.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace reef_perf {

/// The frontend module.
class Frontend : public Module {
  public:
    /// Public port: fetch groups out to the backend (FetchPacket).
    static constexpr const char* kOutInsts = "fetch.ports.out_insts";
    /// Public port: instruction-buffer credits in from the backend.
    static constexpr const char* kInCredits = "fetch.ports.in_credits";

    /// Creates the module; its tree node is top.frontend.
    Frontend();

    /** Registers Fetch's factory.
     *
     *  @param resources The simulation's resource set.
     */
    void add_factories(sparta::ResourceSet& resources) override;

    /// Looks up the fetch unit.
    void bind() override;

    /** Connects the functional simulator. Call after bind().
     *
     *  @param funcsim Functional simulator; must outlive the simulation.
     */
    void set_func_sim(FuncSim* funcsim);

    /** Taken branches, jumps and traps fetched.
     *
     *  @return Redirect count.
     */
    [[nodiscard]] std::uint64_t num_redirects() const;

  protected:
    /** The module's units.
     *
     *  @return {"fetch"}.
     */
    [[nodiscard]] std::vector<std::string> unit_names() const override;

  private:
    /// Fetch unit, valid after bind().
    Fetch* fetch_ = nullptr;
};

} // namespace reef_perf
