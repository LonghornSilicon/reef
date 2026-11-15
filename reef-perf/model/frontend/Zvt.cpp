//===- Zvt.cpp - Zvt (matrix extension) encodings and configuration -------===//

#include "frontend/Zvt.h"

#include <algorithm>
#include <cstdint>
#include <optional>

using namespace reef_perf;

static constexpr uint32_t OpLoadFP = 0x07;
static constexpr uint32_t OpStoreFP = 0x27;
static constexpr uint32_t OpVector = 0x57;  // OP-V
static constexpr uint32_t OpVectorE = 0x77; // OP-VE

static constexpr uint32_t Funct7MSetMType = 0x42; // 1000010
static constexpr uint32_t Funct7MSetT = 0x44;     // 1000100
static constexpr uint32_t Funct6Unary = 0x10;     // 010000
static constexpr uint32_t Funct6MvTV = 0x17;      // 010111
static constexpr uint32_t Funct6MM = 0x3c;        // 111100
static constexpr uint32_t Vs2Zero = 0x1e;         // 11110: vtzero
static constexpr uint32_t Vs2MvVT = 0x1f;         // 11111: vtmv.v.t
static constexpr uint32_t Vs2Discard = 0x1c;      // 11100: vtdiscard

// vtype SEW field values (rvv_backend_define.svh RVVSEW).
static constexpr uint32_t Sew8 = 0;
static constexpr uint32_t Sew16 = 1;
static constexpr uint32_t Sew32 = 2;

/// vtype bit 31 makes Spike set vill (vector_unit.cc: reserved bits).
static constexpr uint32_t SpikeVill = 1U << 31;

static uint32_t getBits(uint32_t Word, int Hi, int Lo) {
  return (Word >> Lo) & ((1U << (Hi - Lo + 1)) - 1U);
}

std::optional<ZvtInst> reef_perf::decodeZvt(uint32_t Word) {
  ZvtInst Z;
  Z.Word = Word;
  Z.Rd = getBits(Word, 11, 7);
  Z.Rs1 = getBits(Word, 19, 15);
  Z.Rs2 = getBits(Word, 24, 20);
  uint32_t Opcode = getBits(Word, 6, 0);
  uint32_t Funct3 = getBits(Word, 14, 12);
  uint32_t Funct6 = getBits(Word, 31, 26);
  uint32_t Funct7 = getBits(Word, 31, 25);
  bool Vm = getBits(Word, 25, 25) != 0;

  auto Make = [&Z](ZvtKind Kind, const char *Name) {
    Z.Kind = Kind;
    Z.Name = Name;
    return std::optional<ZvtInst>(Z);
  };

  switch (Opcode) {
  case OpVector:
    if (Funct3 == 7) { // RvvDecode.scala:186-199
      if (Funct7 == Funct7MSetMType && Z.Rd == 0)
        return Make(ZvtKind::MSetMType, "msetmtype");
      if (Funct7 == Funct7MSetT && Z.Rs2 == 0)
        return Make(ZvtKind::MSetTN, "msettn");
      if (Funct7 == Funct7MSetT && Z.Rs2 == 1)
        return Make(ZvtKind::MSetTM, "msettm");
      if (Funct7 == Funct7MSetT && Z.Rs2 == 2)
        return Make(ZvtKind::MSetTK, "msettk");
      if (getBits(Word, 31, 29) == 4 && getBits(Word, 27, 25) == 4 &&
          getBits(Word, 22, 20) == 4 && getBits(Word, 19, 17) == 0 && Z.Rd == 0)
        return Make(ZvtKind::MSetMTypeI, "msetmtypei");
      return std::nullopt;
    }
    if (Funct3 == 6 && Vm) { // OPMVX; RvvDecode.scala:258-298
      if (Funct6 == Funct6Unary && Z.Rs2 == Vs2Zero && Z.Rs1 == 0)
        return Make(ZvtKind::VtZero, "vtzero");
      if (Funct6 == Funct6Unary && Z.Rs2 == Vs2MvVT)
        return Make(ZvtKind::VtMvVT, "vtmv.v.t");
      if (Funct6 == Funct6Unary && Z.Rs2 == Vs2Discard && Z.Rs1 == 0 &&
          Z.Rd == 0)
        return Make(ZvtKind::VtDiscard, "vtdiscard");
      if (Funct6 == Funct6MvTV && Z.Rd == 0)
        return Make(ZvtKind::VtMvTV, "vtmv.t.v");
    }
    return std::nullopt;
  case OpVectorE: // RvvDecode.scala:411-424
    if (Funct6 != Funct6MM || !Vm || (Funct3 != 0 && Funct3 != 1))
      return std::nullopt;
    Z.Float = Funct3 == 1;
    if (Z.Float)
      return Make(ZvtKind::VtMM, "vtfmm.tvv");
    return Make(ZvtKind::VtMM, (Z.Rd & 1U) ? "vtmms.tvv" : "vtmmu.tvv");
  case OpLoadFP:
  case OpStoreFP: {
    // mew = 1, width 111, rd/vs3 = 0, vm = 1, mop = 00, nf = 0/1/2
    // (RvvDecode.scala:362-376).
    uint32_t NF = getBits(Word, 31, 29);
    if (getBits(Word, 28, 28) != 1 || Funct3 != 7 || Z.Rd != 0 || !Vm ||
        getBits(Word, 27, 26) != 0 || NF > 2)
      return std::nullopt;
    Z.EEWBytes = static_cast<uint8_t>(1U << NF);
    static constexpr const char *Loads[] = {"vtle8", "vtle16", "vtle32"};
    static constexpr const char *Stores[] = {"vtse8", "vtse16", "vtse32"};
    if (Opcode == OpLoadFP)
      return Make(ZvtKind::VtLoad, Loads[NF]);
    return Make(ZvtKind::VtStore, Stores[NF]);
  }
  default:
    return std::nullopt;
  }
}

bool reef_perf::isStandardVset(uint32_t Word) {
  return getBits(Word, 6, 0) == OpVector && getBits(Word, 14, 12) == 7 &&
         !decodeZvt(Word);
}

/// \returns KMAX for SEW field \p Sew (RvvFrontEnd.sv:283-298, :374-379).
static uint32_t getKMax(uint32_t Sew) {
  switch (Sew) {
  case Sew8:
    return 4;
  case Sew16:
    return 2;
  case Sew32:
    return 1;
  default:
    return 0;
  }
}

/// \returns true if SEW field \p Sew with \p MTWiden gives tile elements
/// wider than 32 bits, or \p AltFmt is set at SEW32 or more
/// (RvvFrontEnd.sv:429-443).
static bool isIllegalTileConfig(uint32_t Sew, uint32_t MTWiden, bool AltFmt) {
  if (AltFmt && Sew >= Sew32)
    return true;
  if (MTWiden == 0)
    return false;
  return (Sew == Sew16 && MTWiden == 3) || (Sew == Sew32 && MTWiden >= 2) ||
         Sew > Sew32;
}

/// \returns the vtype Spike gets for a configured matrix unit: LMUL derived
/// from SEW, ta = ma = 1 (RvvFrontEnd.sv:272-281), or vill for a reserved
/// SEW.
static uint32_t getConfiguredVtype(uint32_t Sew) {
  if (Sew > Sew32)
    return SpikeVill;
  // LMUL field: SEW8 -> m1 (000), SEW16 -> m2 (001), SEW32 -> m4 (010).
  return Sew | (Sew << 3) | (1U << 6) | (1U << 7);
}

ZvtConfigEffect reef_perf::applyZvtConfig(const ZvtInst &Z,
                                          const ZvtConfig &Current,
                                          uint32_t SEWBits, uint32_t Rs1Val,
                                          uint32_t Rs2Val) {
  ZvtConfigEffect E;
  E.Next = Current;
  switch (Z.Kind) {
  case ZvtKind::MSetMType: { // RvvFrontEnd.sv:259-312
    uint32_t MTWiden = Rs1Val & 3U;
    uint32_t Sew = getBits(Rs2Val, 5, 3);
    bool AltFmt = getBits(Rs2Val, 8, 8) != 0;
    bool Vill = getBits(Rs2Val, 31, 9) != 0;
    ZvtConfig Next;
    if (MTWiden != 0) {
      Vill = Vill || getBits(Rs1Val, 31, 24) != 0 ||
             getBits(Rs1Val, 9, 8) != 0 || getBits(Rs1Val, 4, 2) != 0;
      Next.MTWiden = MTWiden;
      Next.TK = std::min(getBits(Rs1Val, 7, 5), getKMax(Sew));
      Next.TM = std::min(getBits(Rs1Val, 23, 10), TileEdge);
      E.SpikeVtype = getConfiguredVtype(Sew);
    } else {
      E.SpikeVtype = Rs2Val & ~(1U << 8); // As vsetvl, without altfmt.
    }
    Next.AltFmt = AltFmt;
    Vill = Vill || isIllegalTileConfig(Sew, MTWiden, AltFmt);
    if (Vill) {
      Next = {};
      E.SpikeVtype = SpikeVill;
    }
    E.Next = Next;
    E.WritesVtype = true;
    E.SetsVl = true; // avl = 0, so vl = 0.
    return E;
  }
  case ZvtKind::MSetMTypeI: { // RvvFrontEnd.sv:316-341
    // Instruction bits 16:15 mtwiden, 24:23 SEW, 28 altfmt; tm = tk = 0.
    uint32_t MTWiden = getBits(Z.Word, 16, 15);
    uint32_t Sew = getBits(Z.Word, 24, 23);
    bool AltFmt = getBits(Z.Word, 28, 28) != 0;
    ZvtConfig Next;
    Next.MTWiden = MTWiden;
    Next.AltFmt = AltFmt;
    if (MTWiden != 0)
      E.SpikeVtype = getConfiguredVtype(Sew);
    else
      E.SpikeVtype = Sew << 3; // LMUL1, ta = ma = 0.
    if (isIllegalTileConfig(Sew, MTWiden, AltFmt)) {
      Next = {};
      E.SpikeVtype = SpikeVill;
    }
    E.Next = Next;
    E.WritesVtype = true;
    // avl keeps its default 0, so vl = 0 (RvvFrontEnd.sv:175, :474-480).
    E.SetsVl = true;
    return E;
  }
  case ZvtKind::MSetTN: // RvvFrontEnd.sv:345-349, :425-427
    E.SetsVl = true;
    E.AVL = Rs1Val;
    E.RdIsVl = true;
    return E;
  case ZvtKind::MSetTM: // RvvFrontEnd.sv:350-364
    if (Current.MTWiden == 0) {
      E.Illegal = true;
      return E;
    }
    E.Next.TM = std::min(Rs1Val, TileEdge);
    E.WritesRd = true;
    E.RdValue = E.Next.TM;
    return E;
  case ZvtKind::MSetTK: { // RvvFrontEnd.sv:365-386
    if (Current.MTWiden == 0) {
      E.Illegal = true;
      return E;
    }
    uint32_t Sew = SEWBits == 8 ? Sew8 : SEWBits == 16 ? Sew16 : Sew32;
    E.Next.TK = std::min(Rs1Val, getKMax(Sew));
    E.WritesRd = true;
    E.RdValue = E.Next.TK;
    return E;
  }
  default:
    return E;
  }
}
