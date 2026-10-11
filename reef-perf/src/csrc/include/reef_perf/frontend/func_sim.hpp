#pragma once

/** @file
 *  @brief The functional simulator as the timing model sees it.
 *
 *  Fetch calls FuncSim::next() for every instruction it fetches
 *  (execute-at-fetch). The instruction is executed immediately in Spike, and
 *  next() returns an Inst that already knows its registers, memory addresses
 *  and branch outcome. The timing model only decides *when* each instruction
 *  happens, never *whether* it happens.
 */

#include "reef_perf/common/inst.hpp"
#include "reef_perf/frontend/spike_driver.hpp"

#include <cstdint>
#include <string>

namespace reef_perf {

/// Produces decoded instructions from Spike, in program order.
class FuncSim {
  public:
    /** Loads a program.
     *
     *  @param elf_path RISC-V ELF to run.
     *  @param options Machine configuration.
     *  @throws std::runtime_error if the ELF cannot be loaded.
     */
    FuncSim(const std::string& elf_path, const SpikeOptions& options);

    /** Executes and decodes the next instruction.
     *
     *  @return The instruction, or nullptr once the program has halted.
     *  @throws std::runtime_error if the instruction traps.
     */
    InstPtr next();

    /** Whether the program has halted.
     *
     *  @return True once next() has returned nullptr.
     */
    [[nodiscard]] bool halted() const { return halted_; }

    /** Number of instructions handed to the timing model.
     *
     *  @return Instruction count.
     */
    [[nodiscard]] std::uint64_t executed() const { return executed_; }

  private:
    /// The Spike instance running the program.
    SpikeDriver driver_;
    /// Whether next() has returned nullptr.
    bool halted_ = false;
    /// Instructions handed out so far.
    std::uint64_t executed_ = 0;
};

} // namespace reef_perf
