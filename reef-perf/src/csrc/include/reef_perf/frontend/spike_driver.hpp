#pragma once

/** @file
 *  @brief Execute-at-fetch driver around the Spike instruction-set simulator.
 *
 *  The timing model calls SpikeDriver::step() once per instruction it
 *  fetches. Spike executes that instruction immediately, and step() reports
 *  everything the timing model needs to know about it: PC, encoding,
 *  disassembly, branch outcome, memory accesses and vector configuration.
 *
 *  Reef is in-order and never executes wrong-path instructions, so Spike never
 *  has to be rolled back.
 */

#include "reef_perf/common/inst.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace reef_perf {

/// Encoding of `mpause`, the instruction that ends a program on Reef.
constexpr std::uint32_t kMpauseEncoding = 0x08000073;

/// ISA string for Reef's M3 baseline: RV32IMF + Zbb + Zve32f, VLEN = 128.
constexpr const char* kDefaultIsa = "rv32imf_zicsr_zifencei_zbb_zve32f_zvl128b";

/// A contiguous range of physical memory backed by host RAM.
struct MemoryRegion {
    /// First byte address of the region.
    std::uint32_t start = 0;
    /// Size of the region in bytes.
    std::uint32_t length = 0;
};

/// Configuration of the simulated machine.
struct SpikeOptions {
    /// ISA string passed to Spike.
    std::string isa = kDefaultIsa;
    /// Memory regions. Accesses outside them raise access faults.
    /// The default is Reef's M3 default map: 8 KB ITCM, 32 KB DTCM.
    std::vector<MemoryRegion> regions = {{0x00000000, 0x2000},
                                         {0x00010000, 0x8000}};
};

/// Everything the timing model needs to know about one executed instruction.
struct InstRecord {
    /// Address of the instruction.
    std::uint32_t pc = 0;
    /// Address of the next instruction executed.
    std::uint32_t next_pc = 0;
    /// Raw 32-bit instruction word.
    std::uint32_t encoding = 0;
    /// Spike's disassembly, e.g. "addi a0, a0, 1".
    std::string disasm;
    /// Vector length in effect before the instruction executed.
    std::uint32_t vl = 0;
    /// Selected element width in bytes.
    std::uint8_t sew_bytes = 1;
    /// LMUL times eight (LMUL=1 is 8).
    std::uint8_t lmul8 = 8;
    /// Memory accesses in the order Spike performed them.
    std::vector<MemAccess> mem;
};

/// Runs a RISC-V ELF in Spike one instruction at a time.
class SpikeDriver {
  public:
    /** Loads an ELF and prepares to execute it from its entry point.
     *
     *  @param elf_path Path to a 32-bit RISC-V ELF.
     *  @param options Machine configuration.
     *  @throws std::runtime_error if the ELF cannot be loaded.
     */
    SpikeDriver(const std::string& elf_path, const SpikeOptions& options);

    /// Releases Spike and the simulated memory.
    ~SpikeDriver();

    /// Not copyable: owns a simulator instance.
    SpikeDriver(const SpikeDriver&) = delete;
    /** Not copy-assignable.
     *  @return Never returns.
     */
    SpikeDriver& operator=(const SpikeDriver&) = delete;
    /// Movable.
    SpikeDriver(SpikeDriver&&) noexcept;
    /** Move-assignable.
     *  @return This driver.
     */
    SpikeDriver& operator=(SpikeDriver&&) noexcept;

    /** Executes the next instruction.
     *
     *  `mpause` is reported but not executed, and ends the program.
     *
     *  @return The instruction's record, or std::nullopt once halted.
     *  @throws std::runtime_error if the instruction traps (an illegal
     *          instruction or an access outside every memory region).
     */
    std::optional<InstRecord> step();

    /** Whether the program has reached `mpause`.
     *
     *  @return True once halted.
     */
    [[nodiscard]] bool halted() const;

    /** Number of instructions reported so far, including `mpause`.
     *
     *  @return Instruction count.
     */
    [[nodiscard]] std::uint64_t executed() const;

  private:
    /// Spike state, kept out of this header so Spike's headers do not leak.
    class Impl;
    /// Owned implementation.
    std::unique_ptr<Impl> impl_;
};

} // namespace reef_perf
