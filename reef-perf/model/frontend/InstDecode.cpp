//===- InstDecode.cpp - Instruction classification ------------------------===//

#include "frontend/InstDecode.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace reef_perf;

// Major opcodes (instruction bits 6:0).
static constexpr uint32_t OpLui = 0x37;
static constexpr uint32_t OpAuipc = 0x17;
static constexpr uint32_t OpJal = 0x6f;
static constexpr uint32_t OpJalr = 0x67;
static constexpr uint32_t OpBranch = 0x63;
static constexpr uint32_t OpLoad = 0x03;
static constexpr uint32_t OpStore = 0x23;
static constexpr uint32_t OpImm = 0x13;
static constexpr uint32_t OpReg = 0x33;
static constexpr uint32_t OpMiscMem = 0x0f;
static constexpr uint32_t OpSystem = 0x73;
static constexpr uint32_t OpLoadFP = 0x07;
static constexpr uint32_t OpStoreFP = 0x27;
static constexpr uint32_t OpFMAdd = 0x43;
static constexpr uint32_t OpFMSub = 0x47;
static constexpr uint32_t OpFNMSub = 0x4b;
static constexpr uint32_t OpFNMAdd = 0x4f;
static constexpr uint32_t OpFP = 0x53;
static constexpr uint32_t OpVector = 0x57;

static constexpr uint32_t Funct7MulDiv = 0x01; // M extension.
static constexpr uint32_t Funct6Unary = 0x10;  // VWXUNARY0, VRXUNARY0, ...
static constexpr uint32_t VsetvlTop7 = 0x40;   // vsetvl: bits 31:25.

namespace {

/// Which register file an rs1/rs2/rs3 field reads.
enum class File : uint8_t {
  None, ///< The field is not a register source.
  X,    ///< Integer register.
  F     ///< FP register.
};

/// Class and source register files of a scalar or FP instruction.
struct ScalarForm {
  InstClass Class = InstClass::UNKNOWN;
  File Rs1 = File::None; // What bits 19:15 read.
  File Rs2 = File::None; // What bits 24:20 read.
  File Rs3 = File::None; // What bits 31:27 read.
};

} // namespace

/// \returns bits \p Hi..\p Lo (inclusive) of \p Word.
static uint32_t getBits(uint32_t Word, int Hi, int Lo) {
  return (Word >> Lo) & ((1U << (Hi - Lo + 1)) - 1U);
}

/// Appends register \p Reg of \p RegFile to \p Regs, skipping x0.
static void addReg(std::vector<RegId> &Regs, File RegFile, uint32_t Reg) {
  if (RegFile == File::X && Reg != 0)
    Regs.push_back(RegId::x(Reg));
  else if (RegFile == File::F)
    Regs.push_back(RegId::f(Reg));
}

/// Appends \p Count consecutive vector registers starting at \p Base.
static void addVRegs(std::vector<RegId> &Regs, uint32_t Base, uint32_t Count) {
  for (uint32_t K = 0; K < Count && Base + K < RegId::NumPerFile; ++K)
    Regs.push_back(RegId::v(Base + K));
}

/// \returns the form of an OP-FP instruction (major opcode 0x53).
static ScalarForm getOpFPForm(uint32_t Funct5) {
  switch (Funct5) {
  case 0x03: // fdiv.s
    return {.Class = InstClass::FP_DIV, .Rs1 = File::F, .Rs2 = File::F};
  case 0x0b: // fsqrt.s
    return {.Class = InstClass::FP_DIV, .Rs1 = File::F};
  case 0x1a: // fcvt.s.w[u]
  case 0x1e: // fmv.w.x
    return {.Class = InstClass::FP, .Rs1 = File::X};
  case 0x08: // Conversions between FP formats.
  case 0x18: // fcvt.w[u].s
  case 0x1c: // fmv.x.w, fclass.s
    return {.Class = InstClass::FP, .Rs1 = File::F};
  default: // fadd, fsub, fmul, fsgnj*, fmin/fmax, feq/flt/fle
    return {.Class = InstClass::FP, .Rs1 = File::F, .Rs2 = File::F};
  }
}

/// \returns the form of a non-vector instruction, or std::nullopt for vector
/// instructions and unknown opcodes.
static std::optional<ScalarForm> getScalarForm(uint32_t Word) {
  uint32_t Funct3 = getBits(Word, 14, 12);
  // LOAD-FP and STORE-FP widths 1-4 are scalar FP; the rest are vector.
  bool ScalarFPWidth = Funct3 >= 1 && Funct3 <= 4;
  switch (getBits(Word, 6, 0)) {
  case OpLui:
  case OpAuipc:
    return ScalarForm{.Class = InstClass::ALU};
  case OpJal:
    return ScalarForm{.Class = InstClass::JUMP};
  case OpJalr:
    return ScalarForm{.Class = InstClass::JUMP, .Rs1 = File::X};
  case OpBranch:
    return ScalarForm{
        .Class = InstClass::BRANCH, .Rs1 = File::X, .Rs2 = File::X};
  case OpLoad:
    return ScalarForm{.Class = InstClass::LOAD, .Rs1 = File::X};
  case OpStore:
    return ScalarForm{
        .Class = InstClass::STORE, .Rs1 = File::X, .Rs2 = File::X};
  case OpImm:
    return ScalarForm{.Class = InstClass::ALU, .Rs1 = File::X};
  case OpReg:
    if (getBits(Word, 31, 25) == Funct7MulDiv)
      return ScalarForm{.Class = Funct3 < 4 ? InstClass::MUL : InstClass::DIV,
                        .Rs1 = File::X,
                        .Rs2 = File::X};
    return ScalarForm{.Class = InstClass::ALU, .Rs1 = File::X, .Rs2 = File::X};
  case OpMiscMem:
    return ScalarForm{.Class = InstClass::FENCE};
  case OpSystem: // Funct3 0: ecall, ebreak, mret, wfi, mpause.
    if (Funct3 == 0)
      return ScalarForm{.Class = InstClass::SYSTEM};
    // csrrw/s/c read rs1; the immediate forms (Funct3 5-7) do not.
    return ScalarForm{.Class = InstClass::CSR,
                      .Rs1 = Funct3 <= 3 ? File::X : File::None};
  case OpLoadFP:
    if (!ScalarFPWidth)
      return std::nullopt;
    return ScalarForm{.Class = InstClass::FP_LOAD, .Rs1 = File::X};
  case OpStoreFP:
    if (!ScalarFPWidth)
      return std::nullopt;
    return ScalarForm{
        .Class = InstClass::FP_STORE, .Rs1 = File::X, .Rs2 = File::F};
  case OpFMAdd:
  case OpFMSub:
  case OpFNMSub:
  case OpFNMAdd:
    return ScalarForm{
        .Class = InstClass::FP, .Rs1 = File::F, .Rs2 = File::F, .Rs3 = File::F};
  case OpFP:
    return getOpFPForm(getBits(Word, 31, 27));
  default:
    return std::nullopt;
  }
}

/// \returns the class of an OP-V arithmetic instruction, from \p Mnemonic.
static InstClass getVectorArithClass(std::string_view Mnemonic) {
  auto StartsWithAny = [Mnemonic](std::initializer_list<std::string_view> Ps) {
    return std::ranges::any_of(
        Ps, [Mnemonic](auto Prefix) { return Mnemonic.starts_with(Prefix); });
  };
  if (StartsWithAny({"vmv.x.s", "vcpop", "vfirst", "vfmv.f.s"}))
    return InstClass::V_TO_SCALAR;
  if (StartsWithAny({"vred", "vwred", "vfred", "vfwred", "vrgather", "vslide",
                     "vfslide", "vcompress"}))
    return InstClass::V_PERM;
  if (StartsWithAny({"vdiv", "vrem"}))
    return InstClass::V_DIV;
  if (StartsWithAny({"vfdiv", "vfrdiv", "vfsqrt"}))
    return InstClass::V_FDIV;
  if (StartsWithAny({"vmul", "vmacc", "vnmsac", "vmadd", "vnmsub", "vwmul",
                     "vwmacc", "vsmul"}))
    return InstClass::V_MUL;
  if (Mnemonic.starts_with("vf"))
    return InstClass::V_FP;
  return InstClass::V_ALU;
}

/// \returns true if an OP-V instruction is a multiply-accumulate, which also
/// reads its destination.
static bool readsDestination(uint32_t Funct3, uint32_t Funct6) {
  bool IntMAC = (Funct3 == 2 || Funct3 == 6) &&
                (Funct6 == 0x29 || Funct6 == 0x2b || Funct6 == 0x2d ||
                 Funct6 == 0x2f || Funct6 >= 0x3c);
  bool FPMAC = (Funct3 == 1 || Funct3 == 5) &&
               ((Funct6 >= 0x28 && Funct6 <= 0x2f) || Funct6 >= 0x3c);
  return IntMAC || FPMAC;
}

/// Decodes the class and sources of an OP-V instruction (major opcode 0x57).
/// Coarse: every operand spans LMUL registers; narrowing sources and widening
/// accumulators span 2*LMUL.
static void decodeOpV(Inst &I, uint32_t Word) {
  uint32_t Funct3 = getBits(Word, 14, 12);
  uint32_t Funct6 = getBits(Word, 31, 26);
  uint32_t Vs2 = getBits(Word, 24, 20);
  uint32_t Rs1 = getBits(Word, 19, 15);
  uint32_t Rd = getBits(Word, 11, 7);
  std::string_view Mnemonic = I.Mnemonic;

  if (Funct3 == 7) { // vsetvli / vsetivli / vsetvl
    I.Class = InstClass::VSET;
    addReg(I.Srcs, getBits(Word, 31, 30) == 3 ? File::None : File::X, Rs1);
    addReg(I.Srcs, getBits(Word, 31, 25) == VsetvlTop7 ? File::X : File::None,
           Vs2);
    return;
  }
  I.Class = getVectorArithClass(Mnemonic);

  uint32_t EMUL = std::max<uint32_t>(1, I.LMUL8 / 8);
  bool Widening = Mnemonic.starts_with("vw") || Mnemonic.starts_with("vfw");
  bool Narrowing = Mnemonic.starts_with("vn") || Mnemonic.starts_with("vfn");
  uint32_t Vs2Regs = Narrowing ? 2 * EMUL : EMUL;
  bool Unary = Funct6 == Funct6Unary;

  // The first operand: vs1, or the scalar rs1, depending on Funct3.
  switch (Funct3) {
  case 0: // OPIVV
  case 1: // OPFVV
  case 2: // OPMVV
    // vmv.x.s, vfmv.f.s and the other unary ops have no vs1.
    if (!Unary || Funct3 == 0)
      addVRegs(I.Srcs, Rs1, EMUL);
    break;
  case 4: // OPIVX
  case 6: // OPMVX
    addReg(I.Srcs, File::X, Rs1);
    break;
  case 5: // OPFVF
    addReg(I.Srcs, File::F, Rs1);
    break;
  default: // OPIVI: rs1 is an immediate.
    break;
  }
  // vs2, except for scalar-to-vector moves (vmv.s.x, vfmv.s.f).
  if (!Unary || (Funct3 != 5 && Funct3 != 6))
    addVRegs(I.Srcs, Vs2, Vs2Regs);
  if (getBits(Word, 25, 25) == 0) // Masked: reads v0.
    addVRegs(I.Srcs, 0, 1);
  if (readsDestination(Funct3, Funct6))
    addVRegs(I.Srcs, Rd, Widening ? 2 * EMUL : EMUL);
}

/// Decodes the class and sources of a vector load or store (LOAD-FP or
/// STORE-FP with a vector width).
static void decodeVectorMem(Inst &I, uint32_t Word, bool IsStore) {
  I.Class = IsStore ? InstClass::V_STORE : InstClass::V_LOAD;
  uint32_t Mop = getBits(Word, 27, 26);
  addReg(I.Srcs, File::X, getBits(Word, 19, 15));
  if (Mop == 2) // Strided.
    addReg(I.Srcs, File::X, getBits(Word, 24, 20));
  if (Mop == 1 || Mop == 3) // Indexed (coarse: one index register).
    addVRegs(I.Srcs, getBits(Word, 24, 20), 1);
  if (getBits(Word, 25, 25) == 0) // Masked.
    addVRegs(I.Srcs, 0, 1);
  if (IsStore) { // The data: NF segments of LMUL registers.
    uint32_t NF = getBits(Word, 31, 29) + 1;
    addVRegs(I.Srcs, getBits(Word, 11, 7),
             std::max<uint32_t>(1, I.LMUL8 / 8) * NF);
  }
}

void reef_perf::decodeInst(Inst &I) {
  // Mnemonic = first whitespace-separated token of the disassembly.
  I.Mnemonic = I.Disasm.substr(0, I.Disasm.find_first_of(" \t"));

  uint32_t Word = I.Encoding;
  I.Srcs.clear();
  I.Class = InstClass::UNKNOWN;

  if (std::optional<ScalarForm> Form = getScalarForm(Word)) {
    I.Class = Form->Class;
    addReg(I.Srcs, Form->Rs1, getBits(Word, 19, 15));
    addReg(I.Srcs, Form->Rs2, getBits(Word, 24, 20));
    addReg(I.Srcs, Form->Rs3, getBits(Word, 31, 27));
    return;
  }
  switch (getBits(Word, 6, 0)) {
  case OpLoadFP:
    decodeVectorMem(I, Word, /*IsStore=*/false);
    break;
  case OpStoreFP:
    decodeVectorMem(I, Word, /*IsStore=*/true);
    break;
  case OpVector:
    decodeOpV(I, Word);
    break;
  default:
    break;
  }
}
