//===- ZvtTest.cpp - Zvt decode and configuration tests -------------------===//
//
// Instruction words come from the helpers in CoralNPU 8a5e1b31
// tests/cocotb/vme_test/vme_test_utils.h (ZvtWord and the .insn forms).
//
//===----------------------------------------------------------------------===//

#include "frontend/Zvt.h"
#include "common/Inst.h"
#include "frontend/InstDecode.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <ios>
#include <optional>
#include <vector>

using namespace reef_perf;

/// \returns the Zvt instruction \p Word, failing the test if it is not one.
static ZvtInst mustDecode(uint32_t Word) {
  std::optional<ZvtInst> Z = decodeZvt(Word);
  EXPECT_TRUE(Z.has_value());
  return Z.value_or(ZvtInst{});
}

namespace {

TEST(ZvtTest, DecodesMatrixMultiply) {
  ZvtInst U = mustDecode(0xf2880077); // vtmmu mt0, v8, v16
  EXPECT_EQ(U.Kind, ZvtKind::VtMM);
  EXPECT_FALSE(U.Float);
  EXPECT_EQ(U.Rd, 0U);
  EXPECT_EQ(U.Rs2, 8U);
  EXPECT_EQ(U.Rs1, 16U);
  EXPECT_STREQ(U.Name, "vtmmu.tvv");
  EXPECT_STREQ(mustDecode(0xf28800f7).Name, "vtmms.tvv");
  ZvtInst F = mustDecode(0xf2881077); // vtfmm mt0, v8, v16
  EXPECT_TRUE(F.Float);
}

TEST(ZvtTest, DecodesTileStateAndMoves) {
  EXPECT_EQ(mustDecode(0x43e06057).Kind, ZvtKind::VtZero); // vtzero mt0
  EXPECT_EQ(mustDecode(0x43e06457).Rd, 8U);                // vtzero mt4
  EXPECT_EQ(mustDecode(0x43f56257).Kind, ZvtKind::VtMvVT); // vtmv.v.t v4, a0
  EXPECT_EQ(mustDecode(0x5e456057).Kind, ZvtKind::VtMvTV); // vtmv.t.v a0, v4
  EXPECT_EQ(mustDecode(0x43c06057).Kind, ZvtKind::VtDiscard);
}

TEST(ZvtTest, DecodesTileLoadsAndStores) {
  ZvtInst L8 = mustDecode(0x12f77007); // vtle8 (a4), a5
  EXPECT_EQ(L8.Kind, ZvtKind::VtLoad);
  EXPECT_EQ(L8.EEWBytes, 1);
  EXPECT_EQ(L8.Rs1, 14U);
  EXPECT_EQ(L8.Rs2, 15U);
  EXPECT_EQ(mustDecode(0x12f77027).Kind, ZvtKind::VtStore);
  EXPECT_EQ(mustDecode(0x52f77007).EEWBytes, 4); // vtle32
  // nf = 3 (64-bit) is rejected by the RTL decoder.
  EXPECT_FALSE(decodeZvt(0xd2f77007).has_value());
}

TEST(ZvtTest, DecodesConfiguration) {
  EXPECT_EQ(mustDecode(0x84b57057).Kind, ZvtKind::MSetMType);
  EXPECT_EQ(mustDecode(0x880572d7).Kind, ZvtKind::MSetTN);
  EXPECT_EQ(mustDecode(0x881572d7).Kind, ZvtKind::MSetTM);
  EXPECT_EQ(mustDecode(0x882572d7).Kind, ZvtKind::MSetTK);
  EXPECT_EQ(mustDecode(0x8841f057).Kind, ZvtKind::MSetMTypeI);
  EXPECT_EQ(mustDecode(0x9841f057).Kind, ZvtKind::MSetMTypeI); // altfmt
}

TEST(ZvtTest, LeavesStandardVectorInstructionsAlone) {
  EXPECT_FALSE(decodeZvt(0x0d06f2d7).has_value()); // vsetvli t0, a3, e32
  EXPECT_FALSE(decodeZvt(0x80b572d7).has_value()); // vsetvl t0, a0, a1
  EXPECT_FALSE(decodeZvt(0x420560d7).has_value()); // vmv.s.x v1, a0
  EXPECT_FALSE(decodeZvt(0x02060407).has_value()); // vle8.v v8, (a2)
  EXPECT_TRUE(isStandardVset(0x0d06f2d7));
  EXPECT_TRUE(isStandardVset(0x80b572d7));
  EXPECT_FALSE(isStandardVset(0x84b57057)); // msetmtype
}

TEST(ZvtTest, MSetMTypeConfiguresAndClamps) {
  ZvtInst Z = mustDecode(0x84b57057);
  // tm = 20 (clamped to 16), tk = 7 (clamped to KMAX 4 at SEW8), x4.
  uint32_t MType = (20U << 10) | (7U << 5) | 3U;
  ZvtConfigEffect E = applyZvtConfig(Z, {}, 8, MType, /*SEW8*/ 0);
  EXPECT_FALSE(E.Illegal);
  EXPECT_EQ(E.Next.MTWiden, 3U);
  EXPECT_EQ(E.Next.TM, 16U);
  EXPECT_EQ(E.Next.TK, 4U);
  EXPECT_TRUE(E.WritesVtype);
  EXPECT_EQ(E.SpikeVtype, 0xc0U); // SEW8, LMUL1, ta, ma
  EXPECT_TRUE(E.SetsVl);
  EXPECT_EQ(E.AVL, 0U);

  // SEW32 derives LMUL4 and KMAX 1.
  ZvtConfigEffect E32 =
      applyZvtConfig(Z, {}, 8, (16U << 10) | (4U << 5) | 1U, 2U << 3);
  EXPECT_EQ(E32.SpikeVtype, 0xd2U);
  EXPECT_EQ(E32.Next.TK, 1U);
}

TEST(ZvtTest, MSetMTypeIllegalConfigsClearMType) {
  ZvtInst Z = mustDecode(0x84b57057);
  // SEW16 with x4 widening: 64-bit tile elements.
  ZvtConfigEffect Wide = applyZvtConfig(Z, {}, 8, (16U << 10) | 3U, 1U << 3);
  EXPECT_EQ(Wide.Next.MTWiden, 0U);
  EXPECT_NE(Wide.SpikeVtype & (1U << 31), 0U);
  // altfmt at SEW32.
  ZvtConfigEffect Alt =
      applyZvtConfig(Z, {}, 8, (16U << 10) | 1U, (2U << 3) | (1U << 8));
  EXPECT_NE(Alt.SpikeVtype & (1U << 31), 0U);
  // A reserved rs1 bit with mtwiden != 0.
  ZvtConfigEffect Res = applyZvtConfig(Z, {}, 8, (1U << 30) | 3U, 0);
  EXPECT_NE(Res.SpikeVtype & (1U << 31), 0U);
}

TEST(ZvtTest, MSetMTypeUnconfiguredActsAsVsetvl) {
  ZvtInst Z = mustDecode(0x84b57057);
  // mtwiden = 0: vtype is rs2 without altfmt; mtype is 0. altfmt is legal
  // at SEW8 (and SEW16) but not at SEW32 (RvvFrontEnd.sv:429-433).
  ZvtConfigEffect E = applyZvtConfig(Z, {}, 8, 0, 0xc0U | (1U << 8));
  EXPECT_EQ(E.Next.MTWiden, 0U);
  EXPECT_EQ(E.Next.TM, 0U);
  EXPECT_TRUE(E.Next.AltFmt);
  EXPECT_EQ(E.SpikeVtype, 0xc0U);
  ZvtConfigEffect E32 = applyZvtConfig(Z, {}, 8, 0, 0xd0U | (1U << 8));
  EXPECT_NE(E32.SpikeVtype & (1U << 31), 0U);
}

TEST(ZvtTest, MSetTMAndTKClampAndNeedMType) {
  ZvtConfig Configured{3, 16, 4, false};
  ZvtConfigEffect TM =
      applyZvtConfig(mustDecode(0x881572d7), Configured, 8, 40, 0);
  EXPECT_TRUE(TM.WritesRd);
  EXPECT_EQ(TM.RdValue, 16U);
  ZvtConfigEffect TK =
      applyZvtConfig(mustDecode(0x882572d7), Configured, 16, 7, 0);
  EXPECT_EQ(TK.RdValue, 2U); // KMAX at SEW16
  EXPECT_TRUE(applyZvtConfig(mustDecode(0x881572d7), {}, 8, 4, 0).Illegal);
  EXPECT_TRUE(applyZvtConfig(mustDecode(0x882572d7), {}, 8, 4, 0).Illegal);
}

TEST(ZvtTest, MSetTNRequestsVl) {
  ZvtConfigEffect E = applyZvtConfig(mustDecode(0x880572d7), {}, 8, 12, 0);
  EXPECT_TRUE(E.SetsVl);
  EXPECT_EQ(E.AVL, 12U);
  EXPECT_TRUE(E.RdIsVl);
  EXPECT_FALSE(E.WritesVtype);
}

TEST(ZvtTest, MSetMTypeIReadsImmediates) {
  ZvtConfigEffect E = applyZvtConfig(mustDecode(0x8841f057), {}, 8, 0, 0);
  EXPECT_EQ(E.Next.MTWiden, 3U);
  EXPECT_EQ(E.Next.TM, 0U);
  EXPECT_FALSE(E.Next.AltFmt);
  EXPECT_EQ(E.SpikeVtype, 0xc0U);
  EXPECT_TRUE(applyZvtConfig(mustDecode(0x9841f057), {}, 8, 0, 0).Next.AltFmt);
}

TEST(ZvtTest, TileSliceSelector) {
  TileSlice S = getTileSlice((5U << 27) | (1U << 24) | 9U);
  EXPECT_EQ(S.Tile, 5U);
  EXPECT_TRUE(S.Column);
  EXPECT_EQ(S.Index, 9U);
}

TEST(ZvtTest, InstDecodeClassifiesZvt) {
  Inst Mac;
  Mac.Encoding = 0xf2880077;
  Mac.Disasm = "vtmmu.tvv";
  decodeInst(Mac, /*Zvt=*/true);
  EXPECT_EQ(Mac.Class, InstClass::MATRIX);
  EXPECT_EQ(Mac.Srcs,
            (std::vector<RegId>{RegId::v(8), RegId::v(10), RegId::v(12),
                                RegId::v(14), RegId::v(16), RegId::v(18),
                                RegId::v(20), RegId::v(22)}));

  Inst Load;
  Load.Encoding = 0x12f77007;
  Load.Disasm = "vtle8";
  decodeInst(Load, /*Zvt=*/true);
  EXPECT_EQ(Load.Class, InstClass::MATRIX);
  EXPECT_EQ(Load.Srcs, (std::vector<RegId>{RegId::x(14), RegId::x(15)}));

  Inst Cfg;
  Cfg.Encoding = 0x84b57057;
  Cfg.Disasm = "msetmtype";
  decodeInst(Cfg, /*Zvt=*/true);
  EXPECT_EQ(Cfg.Class, InstClass::VSET);
  EXPECT_EQ(Cfg.Srcs, (std::vector<RegId>{RegId::x(10), RegId::x(11)}));
}

TEST(ZvtTest, InstDecodeIgnoresZvtWhenOff) {
  // vtmmu and vtle8 are MATRIX only with Zvt on.
  for (uint32_t Word : {0xf2880077U, 0x12f77007U}) {
    Inst Off;
    Off.Encoding = Word;
    decodeInst(Off);
    EXPECT_NE(Off.Class, InstClass::MATRIX) << std::hex << Word;
    Inst On;
    On.Encoding = Word;
    decodeInst(On, /*Zvt=*/true);
    EXPECT_EQ(On.Class, InstClass::MATRIX) << std::hex << Word;
  }
}

} // namespace
