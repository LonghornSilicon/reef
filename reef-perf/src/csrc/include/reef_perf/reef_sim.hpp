#pragma once

/** @file
 *  @brief Top-level Sparta simulation: builds the modules, binds the ports
 *         between them and reports the results.
 *
 *  Tree (parameter paths are top.MODULE.UNIT.params.NAME):
 *  - top.frontend  Frontend (fetch, Spike, decode)
 *  - top.backend   Backend (dispatch, scalar_exec, lsu, rob)
 *  - top.vector    Vector (vxu)
 *  - top.mem       Memory (tcm, axi)
 *
 *  This file only knows each module's *public* ports and summary
 *  accessors; everything inside a module belongs to that module. See
 *  docs/interfaces.md.
 */

#include "reef_perf/backend/backend.hpp"
#include "reef_perf/common/module.hpp"
#include "reef_perf/frontend/frontend.hpp"
#include "reef_perf/frontend/func_sim.hpp"
#include "reef_perf/frontend/spike_driver.hpp"
#include "reef_perf/mem/memory.hpp"
#include "reef_perf/vector/vector.hpp"

#include "sparta/app/Simulation.hpp"

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace reef_perf {

/// The Reef simulation.
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
    /// Sparta hook: creates every module's subtree.
    void buildTree_() override;
    /// Sparta hook: nothing to configure beyond the parameters.
    void configureTree_() override;
    /// Sparta hook: binds each module, then the ports between modules.
    void bindTree_() override;

    /** Every module, in build order.
     *
     *  @return Non-owning pointers to the modules.
     */
    [[nodiscard]] std::vector<Module*> modules();

    /** Every module, in reporting order.
     *
     *  @return Non-owning pointers to the modules.
     */
    [[nodiscard]] std::vector<const Module*> modules() const;

    /** Binds a public port of one module to a public port of another.
     *
     *  @param from Module that owns the first port.
     *  @param from_port The first port, relative to its module.
     *  @param to Module that owns the second port.
     *  @param to_port The second port, relative to its module.
     */
    void bind_ports(const Module& from, const char* from_port,
                    const Module& to, const char* to_port);

    /// Functional simulator feeding the frontend.
    std::unique_ptr<FuncSim> funcsim_;
    /// Path of the program being simulated.
    std::string elf_path_;
    /// Frontend module.
    Frontend frontend_;
    /// Backend module.
    Backend backend_;
    /// Vector module.
    Vector vector_;
    /// Memory module.
    Memory mem_;
};

} // namespace reef_perf
