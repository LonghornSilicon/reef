//===- FuncSim.cpp - Functional simulator (Spike) -------------------------===//

#include "frontend/FuncSim.h"
#include "frontend/InstDecode.h"

#include <riscv/decode.h>
#include <riscv/disasm.h>

#include <elf.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace reef_perf;

static constexpr uint16_t ElfMachineRISCV = 243; // EM_RISCV

static std::string toHex(uint64_t Value) {
  std::ostringstream OS;
  OS << "0x" << std::hex << Value;
  return OS.str();
}

/// Copies a POD value out of \p Data at \p Offset, with bounds checking.
template <typename T>
static T readPOD(const std::vector<char> &Data, size_t Offset) {
  if (Offset + sizeof(T) > Data.size())
    throw std::runtime_error("truncated ELF file");
  T Value{};
  std::memcpy(&Value, Data.data() + Offset, sizeof(T));
  return Value;
}

/// Loads the PT_LOAD segments of the 32-bit RISC-V ELF \p Path into \p Mem.
/// \returns the entry point.
static uint32_t loadELF(const std::string &Path, simif_t &Mem) {
  std::ifstream In(Path, std::ios::binary);
  if (!In)
    throw std::runtime_error("cannot open ELF: " + Path);
  const std::vector<char> Data((std::istreambuf_iterator<char>(In)),
                               std::istreambuf_iterator<char>());
  auto Header = readPOD<Elf32_Ehdr>(Data, 0);
  if (std::memcmp(std::data(Header.e_ident), ELFMAG, SELFMAG) != 0 ||
      Header.e_ident[EI_CLASS] != ELFCLASS32 ||
      Header.e_machine != ElfMachineRISCV)
    throw std::runtime_error("not a 32-bit RISC-V ELF: " + Path);
  for (size_t I = 0; I < Header.e_phnum; ++I) {
    auto Segment =
        readPOD<Elf32_Phdr>(Data, Header.e_phoff + (I * Header.e_phentsize));
    if (Segment.p_type != PT_LOAD || Segment.p_memsz == 0)
      continue;
    char *Dst = Mem.addr_to_mem(Segment.p_paddr);
    const char *Last = Mem.addr_to_mem(Segment.p_paddr + Segment.p_memsz - 1);
    if (!Dst || Last != Dst + Segment.p_memsz - 1)
      throw std::runtime_error("ELF segment at " + toHex(Segment.p_paddr) +
                               " is outside the memory map");
    if (Segment.p_offset + Segment.p_filesz > Data.size())
      throw std::runtime_error("truncated ELF segment");
    std::memcpy(Dst, Data.data() + Segment.p_offset, Segment.p_filesz);
    std::memset(Dst + Segment.p_filesz, 0, Segment.p_memsz - Segment.p_filesz);
  }
  return Header.e_entry;
}

/// Converts Spike's register-write log to registers, in ascending order. Spike
/// keys
/// each entry as (register << 4) | type: 0 integer, 1 FP, 2 vector, 3 vector
/// hint, 4 CSR (riscv/decode_macros.h). x0, hints and CSRs are dropped.
static std::vector<RegId> getWrittenRegs(const commit_log_reg_t &Log) {
  constexpr reg_t TypeX = 0;
  constexpr reg_t TypeF = 1;
  constexpr reg_t TypeV = 2;
  std::vector<RegId> Regs;
  for (const auto &Entry : Log) {
    reg_t Type = Entry.first & 0xf;
    reg_t Reg = Entry.first >> 4;
    if (Type == TypeX && Reg != 0)
      Regs.push_back(RegId::x(Reg));
    else if (Type == TypeF)
      Regs.push_back(RegId::f(Reg));
    else if (Type == TypeV)
      Regs.push_back(RegId::v(Reg));
  }
  std::ranges::sort(Regs); // The log is ordered by Spike's key.
  return Regs;
}

FuncSim::FlatMemory::FlatMemory(const std::vector<MemoryRegion> &Regions,
                                const cfg_t *Config)
    : Config(Config) {
  for (const MemoryRegion &Region : Regions)
    Backings.push_back({Region, std::vector<char>(Region.Length, 0)});
}

char *FuncSim::FlatMemory::addr_to_mem(reg_t PAddr) {
  for (Backing &B : Backings) {
    reg_t Start = B.Region.Start;
    if (PAddr >= Start && PAddr - Start < B.Region.Length)
      return &B.Bytes.at(PAddr - Start);
  }
  return nullptr;
}

FuncSim::FuncSim(const std::string &ElfPath, const SpikeOptions &Options)
    : ISA(Options.ISA), Mem(Options.Regions, &Config),
      Log(std::fopen("/dev/null", "w"), &std::fclose) {
  if (!Log)
    throw std::runtime_error("cannot open /dev/null");
  Config.isa = ISA.c_str();
  Config.priv = "M";
  Config.hartids = {0};
  Proc = std::make_unique<processor_t>(ISA.c_str(), "M", &Config, &Mem, 0,
                                       false, Log.get(), ConsoleOut);
  Mem.setHart(Proc.get());
  // Commit logging makes Spike record each instruction's memory accesses and
  // register writes; the textual log goes to /dev/null.
  Proc->enable_log_commits();
  Proc->get_state()->pc = loadELF(ElfPath, Mem);
}

InstRef FuncSim::next() {
  if (Halted)
    return nullptr;
  state_t *State = Proc->get_state();
  reg_t PC = State->pc;
  const char *Host = Mem.addr_to_mem(PC);
  if (!Host)
    throw std::runtime_error("fetch from unmapped address " + toHex(PC));

  auto I = std::make_shared<Inst>();
  I->Seq = NumExecuted++;
  I->PC = static_cast<uint32_t>(PC);
  std::memcpy(&I->Encoding, Host, sizeof(I->Encoding));
  I->Disasm = Proc->get_disassembler()->disassemble(I->Encoding);
  // Before the first vsetvli, vtype is illegal (vill); keep the defaults
  // (vl = 0, SEW = 8, LMUL = 1) rather than garbage.
  if (Proc->VU.vl && !Proc->VU.vill) {
    I->VL = static_cast<uint32_t>(Proc->VU.vl->read());
    I->SEWBytes = static_cast<uint8_t>(Proc->VU.vsew / 8);
    I->LMUL8 = static_cast<uint8_t>(Proc->VU.vflmul * 8);
  }

  if (I->Encoding == MpauseEncoding) {
    // Spike does not implement mpause: report it without executing it.
    Halted = true;
    I->NextPC = I->PC + 4;
  } else {
    execute(*I);
  }
  decodeInst(*I);
  return I;
}

void FuncSim::execute(Inst &I) {
  state_t *State = Proc->get_state();
  reg_t CauseBefore = State->mcause->read();
  reg_t EPCBefore = State->mepc->read();
  State->log_mem_read.clear();
  State->log_mem_write.clear();
  State->log_reg_write.clear();
  // ADDED for DIV
  uint32_t Rs1 = (I.Encoding >> 15) & 0x1f;
  uint32_t Rs2 = (I.Encoding >> 20) & 0x1f;
  I.Rs1Val = static_cast<uint32_t>(State->XPR[Rs1]);
  I.Rs2Val = static_cast<uint32_t>(State->XPR[Rs2]);
  Proc->step(1);
  if (State->mcause->read() != CauseBefore || State->mepc->read() != EPCBefore)
    throw std::runtime_error("trap at " + toHex(I.PC) + " (" + I.Disasm +
                             "), mcause=" + toHex(State->mcause->read()));
  I.NextPC = static_cast<uint32_t>(State->pc);
  for (const auto &[Addr, Value, Size] : State->log_mem_read)
    I.Mem.push_back({static_cast<uint32_t>(Addr), Size, false});
  for (const auto &[Addr, Value, Size] : State->log_mem_write)
    I.Mem.push_back({static_cast<uint32_t>(Addr), Size, true});
  I.Dsts = getWrittenRegs(State->log_reg_write);
}
