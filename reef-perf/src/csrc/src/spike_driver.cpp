#include "reef_perf/spike_driver.hpp"

#include <riscv/cfg.h>
#include <riscv/decode.h>
#include <riscv/disasm.h>
#include <riscv/processor.h>
#include <riscv/simif.h>

#include <elf.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace reef_perf {

namespace {

/// RISC-V machine number in the ELF header (EM_RISCV).
constexpr std::uint16_t kElfMachineRiscv = 243;

/** Formats a value as 0x-prefixed hexadecimal.
 *
 *  @param value Value to format.
 *  @return The formatted string.
 */
std::string hex(std::uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << value;
    return out.str();
}

/** Spike's view of the Reef memory map: a few flat regions of host RAM.
 *
 *  Everything outside the regions is unmapped, so Spike raises an access
 *  fault there. There is no MMIO.
 */
class FlatMemory final : public simif_t {
  public:
    /** Allocates zero-filled backing storage for each region.
     *
     *  @param regions Regions to back with host memory.
     *  @param cfg Spike configuration returned by get_cfg().
     */
    FlatMemory(const std::vector<MemoryRegion>& regions, const cfg_t* cfg)
        : cfg_(cfg) {
        for (const MemoryRegion& region : regions) {
            backing_.push_back({region, std::vector<char>(region.length, 0)});
        }
    }

    /** Translates a physical address to host memory.
     *
     *  @param paddr Physical address.
     *  @return Host pointer, or nullptr if the address is unmapped.
     */
    char* addr_to_mem(reg_t paddr) override {
        for (Backing& backing : backing_) {
            const reg_t start = backing.region.start;
            if (paddr >= start && paddr - start < backing.region.length) {
                return &backing.bytes.at(paddr - start);
            }
        }
        return nullptr;
    }

    /** Rejects MMIO loads: Reef's model has no devices.
     *
     *  @return Always false, which makes Spike raise an access fault.
     */
    bool mmio_load(reg_t /*paddr*/, std::size_t /*len*/,
                   std::uint8_t* /*bytes*/) override {
        return false;
    }

    /** Rejects MMIO stores: Reef's model has no devices.
     *
     *  @return Always false, which makes Spike raise an access fault.
     */
    bool mmio_store(reg_t /*paddr*/, std::size_t /*len*/,
                    const std::uint8_t* /*bytes*/) override {
        return false;
    }

    /// Nothing to reset outside the processor.
    void proc_reset(unsigned /*id*/) override {}

    /** Returns the Spike configuration.
     *
     *  @return The configuration passed to the constructor.
     */
    [[nodiscard]] const cfg_t& get_cfg() const override { return *cfg_; }

    /** Returns the harts, keyed by hart id.
     *
     *  @return The single Reef hart.
     */
    [[nodiscard]] const std::map<std::size_t, processor_t*>&
    get_harts() const override {
        return harts_;
    }

    /** No symbol table is loaded.
     *
     *  @return Always nullptr.
     */
    const char* get_symbol(std::uint64_t /*paddr*/) override { return nullptr; }

    /** Registers the hart once it has been constructed.
     *
     *  @param proc The processor; must outlive this object's use.
     */
    void set_hart(processor_t* proc) { harts_[0] = proc; }

  private:
    /// One region and its host storage.
    struct Backing {
        /// The simulated address range.
        MemoryRegion region;
        /// Host bytes backing the range.
        std::vector<char> bytes;
    };

    /// Spike configuration, owned by SpikeDriver::Impl.
    const cfg_t* cfg_;
    /// Backing storage for every region.
    std::vector<Backing> backing_;
    /// The harts in the system (just one).
    std::map<std::size_t, processor_t*> harts_;
};

/** Copies a POD value out of a byte buffer, with bounds checking.
 *
 *  @tparam T Type to read.
 *  @param data Buffer.
 *  @param offset Byte offset of the value.
 *  @return The value.
 *  @throws std::runtime_error if the value does not fit in the buffer.
 */
template <typename T>
T read_pod(const std::vector<char>& data, std::size_t offset) {
    if (offset + sizeof(T) > data.size()) {
        throw std::runtime_error("truncated ELF file");
    }
    T value{};
    std::memcpy(&value, data.data() + offset, sizeof(T));
    return value;
}

/** Loads a 32-bit RISC-V ELF's PT_LOAD segments into memory.
 *
 *  @param path ELF file.
 *  @param mem Memory to load into.
 *  @return The ELF entry point.
 *  @throws std::runtime_error on a malformed ELF or an unmapped segment.
 */
std::uint32_t load_elf(const std::string& path, FlatMemory& mem) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open ELF: " + path);
    }
    const std::vector<char> data((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
    const auto ehdr = read_pod<Elf32_Ehdr>(data, 0);
    if (std::memcmp(std::data(ehdr.e_ident), ELFMAG, SELFMAG) != 0 ||
        ehdr.e_ident[EI_CLASS] != ELFCLASS32 ||
        ehdr.e_machine != kElfMachineRiscv) {
        throw std::runtime_error("not a 32-bit RISC-V ELF: " + path);
    }
    for (std::size_t i = 0; i < ehdr.e_phnum; ++i) {
        const auto phdr =
            read_pod<Elf32_Phdr>(data, ehdr.e_phoff + (i * ehdr.e_phentsize));
        if (phdr.p_type != PT_LOAD || phdr.p_memsz == 0) {
            continue;
        }
        char* dst = mem.addr_to_mem(phdr.p_paddr);
        const char* last = mem.addr_to_mem(phdr.p_paddr + phdr.p_memsz - 1);
        if (dst == nullptr || last != dst + phdr.p_memsz - 1) {
            throw std::runtime_error("ELF segment at " + hex(phdr.p_paddr) +
                                     " is outside the memory map");
        }
        if (phdr.p_offset + phdr.p_filesz > data.size()) {
            throw std::runtime_error("truncated ELF segment");
        }
        std::memcpy(dst, data.data() + phdr.p_offset, phdr.p_filesz);
        std::memset(dst + phdr.p_filesz, 0, phdr.p_memsz - phdr.p_filesz);
    }
    return ehdr.e_entry;
}

} // namespace

/// Spike objects behind SpikeDriver.
class SpikeDriver::Impl {
  public:
    /** Builds a single-hart machine and loads the program.
     *
     *  @param elf_path ELF to run.
     *  @param options Machine configuration.
     */
    Impl(const std::string& elf_path, const SpikeOptions& options)
        : isa_(options.isa), mem_(options.regions, &cfg_),
          log_(std::fopen("/dev/null", "w"), &std::fclose) {
        if (!log_) {
            throw std::runtime_error("cannot open /dev/null");
        }
        cfg_.isa = isa_.c_str();
        cfg_.priv = "M";
        cfg_.hartids = {0};
        proc_ = std::make_unique<processor_t>(isa_.c_str(), "M", &cfg_, &mem_,
                                              0, false, log_.get(), sout_);
        mem_.set_hart(proc_.get());
        // Commit logging makes Spike record each instruction's memory
        // accesses; the textual log itself goes to /dev/null.
        proc_->enable_log_commits();
        proc_->get_state()->pc = load_elf(elf_path, mem_);
    }

    /** Executes one instruction; see SpikeDriver::step().
     *
     *  @return The instruction record, or std::nullopt once halted.
     */
    std::optional<InstRecord> step() {
        if (halted_) {
            return std::nullopt;
        }
        state_t* state = proc_->get_state();
        const reg_t pc = state->pc;
        const char* host = mem_.addr_to_mem(pc);
        if (host == nullptr) {
            throw std::runtime_error("fetch from unmapped address " + hex(pc));
        }

        InstRecord rec;
        rec.pc = static_cast<std::uint32_t>(pc);
        std::memcpy(&rec.encoding, host, sizeof(rec.encoding));
        rec.disasm = proc_->get_disassembler()->disassemble(rec.encoding);
        // Before the first vsetvli, vtype is illegal (vill); keep the
        // defaults (vl = 0, SEW = 8, LMUL = 1) rather than garbage.
        if (proc_->VU.vl && !proc_->VU.vill) {
            rec.vl = static_cast<std::uint32_t>(proc_->VU.vl->read());
            rec.sew_bytes = static_cast<std::uint8_t>(proc_->VU.vsew / 8);
            rec.lmul8 = static_cast<std::uint8_t>(proc_->VU.vflmul * 8);
        }
        ++executed_;

        // mpause ends the program. Spike does not implement it, so report it
        // without executing it.
        if (rec.encoding == kMpauseEncoding) {
            halted_ = true;
            rec.next_pc = rec.pc + 4;
            return rec;
        }

        const reg_t cause_before = state->mcause->read();
        const reg_t epc_before = state->mepc->read();
        state->log_mem_read.clear();
        state->log_mem_write.clear();
        proc_->step(1);
        if (state->mcause->read() != cause_before ||
            state->mepc->read() != epc_before) {
            throw std::runtime_error("trap at " + hex(pc) + " (" + rec.disasm +
                                     "), mcause=" + hex(state->mcause->read()));
        }

        rec.next_pc = static_cast<std::uint32_t>(state->pc);
        for (const auto& [addr, value, size] : state->log_mem_read) {
            rec.mem.push_back({static_cast<std::uint32_t>(addr), size, false});
        }
        for (const auto& [addr, value, size] : state->log_mem_write) {
            rec.mem.push_back({static_cast<std::uint32_t>(addr), size, true});
        }
        return rec;
    }

    /** Whether mpause has been reached.
     *
     *  @return True once halted.
     */
    [[nodiscard]] bool halted() const { return halted_; }

    /** Instructions reported so far.
     *
     *  @return Instruction count.
     */
    [[nodiscard]] std::uint64_t executed() const { return executed_; }

  private:
    /// ISA string; cfg_ points into it.
    std::string isa_;
    /// Spike configuration.
    cfg_t cfg_;
    /// Simulated memory.
    FlatMemory mem_;
    /// Destination of Spike's textual commit log.
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> log_;
    /// Destination of Spike's console output.
    std::ostringstream sout_;
    /// The single Reef hart.
    std::unique_ptr<processor_t> proc_;
    /// Instructions reported so far.
    std::uint64_t executed_ = 0;
    /// Whether mpause has been reached.
    bool halted_ = false;
};

SpikeDriver::SpikeDriver(const std::string& elf_path,
                         const SpikeOptions& options)
    : impl_(std::make_unique<Impl>(elf_path, options)) {}

SpikeDriver::~SpikeDriver() = default;
SpikeDriver::SpikeDriver(SpikeDriver&&) noexcept = default;
SpikeDriver& SpikeDriver::operator=(SpikeDriver&&) noexcept = default;

std::optional<InstRecord> SpikeDriver::step() { return impl_->step(); }

bool SpikeDriver::halted() const { return impl_->halted(); }

std::uint64_t SpikeDriver::executed() const { return impl_->executed(); }

} // namespace reef_perf
