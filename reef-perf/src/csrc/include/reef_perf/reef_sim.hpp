#pragma once

/** @file
 *  @brief Top-level Sparta simulation: builds the unit tree, wires the ports
 *         and connects the functional simulator to Fetch.
 *
 *  Tree (parameter paths are top.core.UNIT.params.NAME):
 *  - top.core.fetch    Fetch
 *  - top.core.dispatch Dispatch
 *  - top.core.execute  Execute
 *  - top.core.rob      Rob
 */

#include "reef_perf/func_sim.hpp"
#include "reef_perf/spike_driver.hpp"

#include "sparta/app/Simulation.hpp"

#include <memory>
#include <ostream>
#include <string>

namespace reef_perf {

class Fetch;
class Dispatch;
class Execute;
class Rob;

/// The Reef core simulation.
class ReefSim : public sparta::app::Simulation {
  public:
    /** Creates the simulation and loads the program into Spike.
     *
     *  @param scheduler Sparta scheduler that drives the simulation.
     *  @param elf_path RISC-V ELF to run.
     *  @param options Functional-simulator machine configuration.
     *  @throws std::runtime_error if the ELF cannot be loaded.
     */
    ReefSim(sparta::Scheduler& scheduler, const std::string& elf_path,
            const SpikeOptions& options);

    /// Tears down the Sparta tree.
    ~ReefSim() override;

    /// Not copyable: owns the simulation tree.
    ReefSim(const ReefSim&) = delete;
    /** Not copy-assignable.
     *  @return Never returns.
     */
    ReefSim& operator=(const ReefSim&) = delete;
    /// Not movable: Sparta keeps pointers into the tree.
    ReefSim(ReefSim&&) = delete;
    /** Not move-assignable.
     *  @return Never returns.
     */
    ReefSim& operator=(ReefSim&&) = delete;

    /** Writes the human-readable end-of-run summary.
     *
     *  @param os Stream to write to.
     */
    void print_summary(std::ostream& os) const;

    /** Writes the end-of-run summary as JSON, for scripts.
     *
     *  @param os Stream to write to.
     */
    void write_json(std::ostream& os) const;

  private:
    /// Sparta hook: creates the core and its units.
    void buildTree_() override;
    /// Sparta hook: nothing to configure beyond the parameters.
    void configureTree_() override;
    /// Sparta hook: binds the ports and connects FuncSim to Fetch.
    void bindTree_() override;

    /// Functional simulator feeding Fetch.
    std::unique_ptr<FuncSim> funcsim_;
    /// Path of the program being simulated.
    std::string elf_path_;
    /// Fetch unit, valid after bindTree_().
    Fetch* fetch_ = nullptr;
    /// Dispatch unit, valid after bindTree_().
    Dispatch* dispatch_ = nullptr;
    /// Execute unit, valid after bindTree_().
    Execute* execute_ = nullptr;
    /// Retirement buffer, valid after bindTree_().
    Rob* rob_ = nullptr;
};

} // namespace reef_perf
