//===- Zvt.h - Zvt matrix-extension encodings and configuration -*- C++ -*-===//
//-*-===//
///
/// \file
/// The CoralNPU Zvt matrix extension (spec v0.4, as implemented at CoralNPU
/// 8a5e1b31), which Spike does not know. FuncSim does not hand these
/// instructions to Spike: it applies their configuration effects (mtype, and
/// vtype/vl through Spike's vector unit) itself and treats every other Zvt
/// instruction as opaque. Tile contents are not modeled, so programs whose
/// control flow depends on matrix results run the wrong path.
///
/// Encodings: hdl/chisel/src/coralnpu/rvv/RvvDecode.scala:178-205, 211-307,
/// 411-424. Configuration semantics: hdl/verilog/rvv/design/RvvFrontEnd.sv
/// :200-480. Both paths are in the CoralNPU checkout at 8a5e1b31.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_FRONTEND_ZVT_H
#define REEF_PERF_FRONTEND_ZVT_H

#include <cstdint>
#include <optional>

namespace reef_perf {

/// The Zvt instructions.
enum class ZvtKind : uint8_t {
  MSetMType,  ///< msetmtype rs1, rs2: mtype <- rs1, vtype <- rs2, vl <- 0.
  MSetMTypeI, ///< msetmtypei: mtwiden, SEW and altfmt from immediates.
  MSetTN,     ///< msettn rd, rs1: vl <- min(rs1, VLMAX); rd <- vl.
  MSetTM,     ///< msettm rd, rs1: tm <- min(rs1, 16); rd <- tm.
  MSetTK,     ///< msettk rd, rs1: tk <- min(rs1, KMAX); rd <- tk.
  VtZero,     ///< vtzero mtd.
  VtDiscard,  ///< vtdiscard.
  VtMvTV,     ///< vtmv.t.v rs1, vs2: vector registers to a tile slice.
  VtMvVT,     ///< vtmv.v.t vd, rs1: a tile slice to vector registers.
  VtLoad,     ///< vtle8/16/32 rs1, rs2: memory at rs1 to slice TSS(rs2).
  VtStore,    ///< vtse8/16/32 rs1, rs2: slice TSS(rs2) to memory at rs1.
  VtMM        ///< vtmmu, vtmms, vtfmm mtd, vs2, vs1.
};

/// One decoded Zvt instruction: its kind and raw fields.
struct ZvtInst {
  ZvtKind Kind = ZvtKind::VtZero; ///< Which instruction.
  uint32_t Word = 0;              ///< The instruction word.
  uint32_t Rd = 0;       ///< Bits 11:7 (mtd << 1 | flag for vtzero/vtmm).
  uint32_t Rs1 = 0;      ///< Bits 19:15 (rs1 or vs1).
  uint32_t Rs2 = 0;      ///< Bits 24:20 (rs2 or vs2).
  uint8_t EEWBytes = 0;  ///< Memory element bytes, for VtLoad/VtStore.
  bool Float = false;    ///< vtfmm (funct3 001) rather than vtmmu/vtmms.
  const char *Name = ""; ///< Mnemonic, e.g. "vtmmu.tvv".
};

/// \returns the Zvt instruction encoded by \p Word, or std::nullopt if
/// \p Word is not a Zvt instruction.
std::optional<ZvtInst> decodeZvt(uint32_t Word);

/// \returns true for a standard vsetvli, vsetivli or vsetvl, which also
/// clears mtype (RvvFrontEnd.sv:205, :221, :248).
bool isStandardVset(uint32_t Word);

/// The matrix configuration Spike does not hold: mtype and vtype.altfmt.
struct ZvtConfig {
  uint32_t MTWiden = 0; ///< mtype.mtwiden: 0 unconfigured, 1/2/3 = x1/x2/x4.
  uint32_t TM = 0;      ///< mtype.tm, at most 16.
  uint32_t TK = 0;      ///< mtype.tk, at most KMAX.
  bool AltFmt = false;  ///< vtype.altfmt (bit 8), which Spike rejects.
};

/// What a configuration instruction does, for FuncSim to apply.
struct ZvtConfigEffect {
  ZvtConfig Next;           ///< The configuration afterwards.
  bool Illegal = false;     ///< The RTL raises an illegal-instruction trap.
  bool WritesVtype = false; ///< Spike's vtype must change to SpikeVtype.
  /// vtype to give Spike, altfmt removed. Bit 31 set means vill.
  uint32_t SpikeVtype = 0;
  bool SetsVl = false; ///< Spike's vl must be set from AVL (and VLMAX).
  uint32_t AVL = 0;    ///< Requested vl.
  /// rd receives Spike's new vl (msettn) rather than RdValue.
  bool RdIsVl = false;
  bool WritesRd = false; ///< rd receives RdValue (msettm, msettk).
  uint32_t RdValue = 0;  ///< Value for rd.
};

/// \returns the effect of configuration instruction \p Z (an mset*), run
/// with configuration \p Current, vector SEW \p SEWBits (8, 16 or 32) and
/// register values \p Rs1Val and \p Rs2Val.
ZvtConfigEffect applyZvtConfig(const ZvtInst &Z, const ZvtConfig &Current,
                               uint32_t SEWBits, uint32_t Rs1Val,
                               uint32_t Rs2Val);

/// \returns true for the mset* kinds.
inline bool isZvtConfig(ZvtKind Kind) {
  return Kind == ZvtKind::MSetMType || Kind == ZvtKind::MSetMTypeI ||
         Kind == ZvtKind::MSetTN || Kind == ZvtKind::MSetTM ||
         Kind == ZvtKind::MSetTK;
}

/// Fields of a tile slice selector (TSS) register value
/// (rvv_backend_dispatch.sv:506-508).
struct TileSlice {
  uint32_t Tile = 0;   ///< Bits 30:27.
  bool Column = false; ///< Bit 24: column rather than row.
  uint32_t Index = 0;  ///< Bits 3:0: which row or column.
};

/// \returns the slice selected by TSS value \p Value.
inline TileSlice getTileSlice(uint32_t Value) {
  return {(Value >> 27) & 0xfU, ((Value >> 24) & 1U) != 0, Value & 0xfU};
}

/// Tile edge in elements: VLEN / 8 (rvv_backend_define.svh:208).
inline constexpr uint32_t TileEdge = 16;

/// Architectural tiles mt0-mt15 (rvv_backend_define.svh:203).
inline constexpr uint32_t NumTiles = 16;

} // namespace reef_perf

#endif // REEF_PERF_FRONTEND_ZVT_H
