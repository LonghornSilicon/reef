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
#include <optional>
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
  } else if (std::optional<ZvtInst> Z =
                 ZvtEnabled ? decodeZvt(I->Encoding) : std::nullopt) {
    executeZvt(*I, *Z);
  } else {
    execute(*I);
    if (ZvtEnabled && isStandardVset(I->Encoding))
      Zvt = {}; // vset* also clears mtype (RvvFrontEnd.sv:205).
  }
  decodeInst(*I, ZvtEnabled);
  return I;
}

void FuncSim::execute(Inst &I) {
  state_t *State = Proc->get_state();
  reg_t CauseBefore = State->mcause->read();
  reg_t EPCBefore = State->mepc->read();
  State->log_mem_read.clear();
  State->log_mem_write.clear();
  State->log_reg_write.clear();
  // Capture operands before Spike updates the destination register.
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
    I.Mem.push_back({static_cast<uint32_t>(Addr), Size, true, Value});
  I.Dsts = getWrittenRegs(State->log_reg_write);
}

/// \returns the tile-element bytes for mtype.mtwiden \p MTWiden at SEW
/// \p SEWBytes: SEW x TWIDEN, TWIDEN = 1, 2, 4 (vtzero's group,
/// rvv_backend_decode_unit_ari.sv:2735-2752).
static uint32_t getTileElemBytes(uint32_t SEWBytes, uint32_t MTWiden) {
  return SEWBytes << (MTWiden == 0 ? 0 : MTWiden - 1);
}

/// Fills \p I's tile group: \p Tile rounded down to a group of \p ElemBytes
/// tiles (one tile per element byte; zvt_ctrl.sv forces the base).
static void setTileGroup(Inst &I, uint32_t Tile, uint32_t ElemBytes) {
  uint32_t Count = std::clamp<uint32_t>(ElemBytes, 1, 4);
  I.Mtx.NumTiles = static_cast<uint8_t>(Count);
  I.Mtx.FirstTile = static_cast<uint8_t>(Tile & ~(Count - 1U) & 0xfU);
}

void FuncSim::executeZvt(Inst &I, const ZvtInst &Z) {
  state_t *State = Proc->get_state();
  I.Rs1Val = static_cast<uint32_t>(State->XPR[Z.Rs1]);
  I.Rs2Val = static_cast<uint32_t>(State->XPR[Z.Rs2]);
  I.Disasm = Z.Name; // Spike's disassembler does not know Zvt.
  I.NextPC = I.PC + 4;
  State->pc = I.NextPC; // Spike's minstret does not count Zvt instructions.
  auto Trap = [&I](const std::string &Why) {
    throw std::runtime_error("trap at " + toHex(I.PC) + " (" + I.Disasm +
                             "): " + Why);
  };
  auto WriteRd = [&](uint32_t Value) {
    if (Z.Rd == 0)
      return;
    State->XPR.write(Z.Rd, Value);
    I.Dsts = {RegId::x(Z.Rd)};
  };

  if (isZvtConfig(Z.Kind)) {
    ZvtConfigEffect E =
        applyZvtConfig(Z, Zvt, uint32_t{I.SEWBytes} * 8, I.Rs1Val, I.Rs2Val);
    if (E.Illegal)
      Trap("mtype is unconfigured (mtwiden = 0)");
    reg_t Vtype = E.WritesVtype ? E.SpikeVtype : Proc->VU.vtype->read();
    if (E.SetsVl) {
      // rd = rs1 = 1 only steers set_vl to "vl = min(AVL, VLMAX)".
      reg_t Vl = Proc->VU.set_vl(1, 1, E.AVL, Vtype);
      if (E.RdIsVl)
        WriteRd(static_cast<uint32_t>(Vl));
    }
    if (E.WritesRd)
      WriteRd(E.RdValue);
    Zvt = E.Next;
    return;
  }

  // Every tile instruction traps on vill and on an unconfigured mtype: the
  // arithmetic, moves and vtzero in the RVV front end (RvvFrontEnd.sv
  // :624-633), tile loads and stores in scalar decode (Decode.scala:573-579,
  // :1052-1073). vtdiscard needs only a legal vtype (Decode.scala:601-607).
  if (Proc->VU.vill)
    Trap("vtype is illegal (vill)");
  if (Z.Kind != ZvtKind::VtDiscard && Zvt.MTWiden == 0)
    Trap("mtype is unconfigured (mtwiden = 0)");

  uint32_t VL = I.VL;
  uint32_t EMUL = std::max<uint32_t>(1, I.LMUL8 / 8);
  // Shapes the RVV backend decoder rejects retire as no-ops without reaching
  // the matrix engine (rvv_backend_decode_unit.sv:92-96): vl = 0, misaligned
  // tile groups, vtmm with tm, tn or tk of 0, a mismatched type
  // configuration or misaligned vs1/vs2 (rvv_backend_decode_unit_ari.sv:3072,
  // :3211-3216, :3219-3233,
  // :3317-3320, :3605, :3717-3722; _lsu.sv:3387-3393).
  auto MakeNop = [&I] {
    I.Mtx = {};
    I.Mtx.Op = MatrixOp::Nop;
    I.Mtx.NumTiles = 0;
  };
  auto Misaligned = [](uint32_t Tile, uint32_t Count) {
    return (Tile & (Count - 1U)) != 0;
  };
  switch (Z.Kind) {
  case ZvtKind::VtZero: {
    uint32_t Count = getTileElemBytes(I.SEWBytes, Zvt.MTWiden);
    if ((Z.Rd & 1U) != 0 || Misaligned(Z.Rd >> 1, Count)) {
      MakeNop();
      break;
    }
    I.Mtx.Op = MatrixOp::Zero;
    setTileGroup(I, Z.Rd >> 1, Count);
    break;
  }
  case ZvtKind::VtDiscard:
    I.Mtx.Op = MatrixOp::Discard;
    I.Mtx.NumTiles = static_cast<uint8_t>(NumTiles);
    break;
  case ZvtKind::VtMvTV:
  case ZvtKind::VtMvVT: {
    if (VL == 0) {
      MakeNop();
      break;
    }
    bool ToTile = Z.Kind == ZvtKind::VtMvTV;
    I.Mtx.Op = ToTile ? MatrixOp::MoveToTile : MatrixOp::MoveToVector;
    I.Mtx.ElemBytes = I.SEWBytes;
    I.Mtx.TN = static_cast<uint8_t>(std::min(VL, TileEdge));
    setTileGroup(I, getTileSlice(I.Rs1Val).Tile, I.SEWBytes);
    if (!ToTile) // vd: SEW8/16/32 -> 1/2/4 registers (LMUL 1/2/4).
      for (uint32_t K = 0; K < EMUL && Z.Rd + K < RegId::NumPerFile; ++K)
        I.Dsts.push_back(RegId::v(Z.Rd + K));
    break;
  }
  case ZvtKind::VtLoad:
  case ZvtKind::VtStore: {
    if (VL == 0) {
      MakeNop();
      break;
    }
    bool IsStore = Z.Kind == ZvtKind::VtStore;
    I.Mtx.Op = IsStore ? MatrixOp::TileStore : MatrixOp::TileLoad;
    I.Mtx.ElemBytes = Z.EEWBytes;
    // vl is clamped to the tile edge (Lsu.scala:328); unit stride from rs1.
    uint32_t Elems = std::min(VL, TileEdge);
    I.Mtx.TN = static_cast<uint8_t>(Elems);
    setTileGroup(I, getTileSlice(I.Rs2Val).Tile, Z.EEWBytes);
    for (uint32_t E = 0; E < Elems; ++E)
      I.Mem.push_back({I.Rs1Val + E * Z.EEWBytes, Z.EEWBytes, IsStore});
    break;
  }
  case ZvtKind::VtMM: {
    // Legal configurations (rvv_backend_decode_unit_ari.sv:2472-2492,
    // :3039-3055): int8 {SEW8, x4}; bf16 {SEW16, x2, altfmt}; fp32 {SEW32,
    // x1, no altfmt}.
    bool TypeOk = !Z.Float ? I.SEWBytes == 1 && Zvt.MTWiden == 3
                  : I.SEWBytes == 2
                      ? Zvt.MTWiden == 2 && Zvt.AltFmt
                      : I.SEWBytes == 4 && Zvt.MTWiden == 1 && !Zvt.AltFmt;
    // vs1 and vs2 must be aligned for their element width, or the decoder
    // rejects the vtmm and it retires as a no-op (check_vme_vs{1,2}_align,
    // rvv_backend_decode_unit_ari.sv:3219-3233, :3789-3791): int8 needs
    // register bits 2:1 = 0 (v0, v1, v8, v9, ...), bf16 bits 2 and 0 = 0
    // (v0, v2, v8, v10, ...), fp32 bits 1:0 = 0 (v0, v4, v8, ...).
    auto OperandOk = [&I](uint32_t Reg) {
      switch (I.SEWBytes) {
      case 1:
        return (Reg & 0b110U) == 0;
      case 2:
        return (Reg & 0b101U) == 0;
      default:
        return (Reg & 0b011U) == 0;
      }
    };
    if (!TypeOk || Zvt.TM == 0 || VL == 0 || Zvt.TK == 0 ||
        Misaligned(Z.Rd >> 1, 4) || !OperandOk(Z.Rs1) || !OperandOk(Z.Rs2)) {
      MakeNop();
      break;
    }
    I.Mtx.Op = MatrixOp::Mac;
    I.Mtx.TM = static_cast<uint8_t>(Zvt.TM);
    I.Mtx.TN = static_cast<uint8_t>(std::min(VL, TileEdge));
    I.Mtx.TK = static_cast<uint8_t>(Zvt.TK);
    I.Mtx.Type = !Z.Float          ? MatrixType::Int8
                 : I.SEWBytes == 2 ? MatrixType::Bf16
                                   : MatrixType::Fp32;
    I.Mtx.ElemBytes = 4;
    setTileGroup(I, Z.Rd >> 1, 4); // The accumulator has 32-bit elements.
    break;
  }
  default:
    break;
  }
}
