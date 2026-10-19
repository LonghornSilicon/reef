//===- InstDecodeTest.cpp - Instruction decode tests ----------------------===//

#include "frontend/InstDecode.h"
#include "common/Inst.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace reef_perf;

using Regs = std::vector<RegId>;

/// Decodes \p Encoding / \p Disasm with LMUL x 8 = \p LMUL8.
static Inst decode(uint32_t Encoding, const std::string &Disasm,
                   uint8_t LMUL8 = 8) {
  Inst I;
  I.Encoding = Encoding;
  I.Disasm = Disasm;
  I.LMUL8 = LMUL8;
  decodeInst(I);
  return I;
}

static constexpr RegId x(unsigned Reg) { return RegId::x(Reg); }
static constexpr RegId f(unsigned Reg) { return RegId::f(Reg); }
static constexpr RegId v(unsigned Reg) { return RegId::v(Reg); }

namespace {

TEST(InstDecodeTest, ExtractsMnemonic) {
  EXPECT_EQ(decode(0x00b50533, "add     a0, a0, a1").Mnemonic, "add");
}

TEST(InstDecodeTest, IntegerALUSkipsX0) {
  Inst Li = decode(0x01000513, "li      a0, 16"); // addi a0, zero, 16
  EXPECT_EQ(Li.Class, InstClass::ALU);
  EXPECT_EQ(Li.Srcs, Regs{});

  Inst Add = decode(0x00b50533, "add     a0, a0, a1");
  EXPECT_EQ(Add.Srcs, (Regs{x(10), x(11)}));
}

TEST(InstDecodeTest, MultiplyAndDivide) {
  EXPECT_EQ(decode(0x02a585b3, "mul     a1, a1, a0").Class, InstClass::MUL);
  EXPECT_EQ(decode(0x02c55533, "divu    a0, a0, a2").Class, InstClass::DIV);
}

TEST(InstDecodeTest, LoadsAndStores) {
  Inst Load = decode(0x00052583, "lw      a1, 0(a0)");
  EXPECT_EQ(Load.Class, InstClass::LOAD);
  EXPECT_EQ(Load.Srcs, (Regs{x(10)}));
  EXPECT_TRUE(Load.isMemory());

  Inst Store = decode(0x00c5a023, "sw      a2, 0(a1)");
  EXPECT_EQ(Store.Class, InstClass::STORE);
  EXPECT_EQ(Store.Srcs, (Regs{x(11), x(12)}));
}

TEST(InstDecodeTest, CSRReadOfMcycle) {
  Inst CSR = decode(0xb0002d73, "csrr    s10, mcycle");
  EXPECT_EQ(CSR.Class, InstClass::CSR);
  EXPECT_EQ(CSR.Srcs, Regs{});
}

TEST(InstDecodeTest, MpauseIsSystem) {
  EXPECT_EQ(decode(0x08000073, "mpause").Class, InstClass::SYSTEM);
}

TEST(InstDecodeTest, FloatingPointUsesFRegisters) {
  Inst FAdd = decode(0x003170d3, "fadd.s  ft1, ft2, ft3");
  EXPECT_EQ(FAdd.Class, InstClass::FP);
  EXPECT_EQ(FAdd.Srcs, (Regs{f(2), f(3)}));
}

TEST(InstDecodeTest, VsetvliReadsAVL) {
  Inst Vset = decode(0x0c0572d7, "vsetvli t0, a0, e8, m1, ta, ma");
  EXPECT_EQ(Vset.Class, InstClass::VSET);
  EXPECT_EQ(Vset.Srcs, (Regs{x(10)}));
  EXPECT_TRUE(Vset.isVector());
}

TEST(InstDecodeTest, VectorUnitStrideLoad) {
  Inst Vle = decode(0x02058087, "vle8.v  v1, (a1)");
  EXPECT_EQ(Vle.Class, InstClass::V_LOAD);
  EXPECT_EQ(Vle.Srcs, (Regs{x(11)}));
}

TEST(InstDecodeTest, VectorAddAtLMULOne) {
  Inst Vadd = decode(0x02108157, "vadd.vv v2, v1, v1");
  EXPECT_EQ(Vadd.Class, InstClass::V_ALU);
  EXPECT_EQ(Vadd.Srcs, (Regs{v(1), v(1)}));
}

TEST(InstDecodeTest, VectorAddAtLMULFourUsesRegisterGroups) {
  Inst Vadd = decode(0x02440657, "vadd.vv v12, v4, v8", 32);
  EXPECT_EQ(Vadd.Srcs,
            (Regs{v(8), v(9), v(10), v(11), v(4), v(5), v(6), v(7)}));
}

TEST(InstDecodeTest, MaskedVectorOpReadsV0) {
  Inst Vadd = decode(0x00108157, "vadd.vv v2, v1, v1, v0.t");
  EXPECT_EQ(Vadd.Srcs, (Regs{v(1), v(1), v(0)}));
}

TEST(InstDecodeTest, MultiplyAccumulateReadsDestination) {
  Inst Vmacc = decode(0xb620a1d7, "vmacc.vv v3, v1, v2");
  EXPECT_EQ(Vmacc.Class, InstClass::V_MUL);
  EXPECT_EQ(Vmacc.Srcs, (Regs{v(1), v(2), v(3)}));
}

TEST(InstDecodeTest, ReductionIsPermuteClass) {
  Inst Vred = decode(0x02302257, "vredsum.vs v4, v3, v0");
  EXPECT_EQ(Vred.Class, InstClass::V_PERM);
  EXPECT_EQ(Vred.Srcs, (Regs{v(0), v(3)}));
}

TEST(InstDecodeTest, VmvXsIsVectorToScalar) {
  Inst Vmv = decode(0x42402657, "vmv.x.s a2, v4");
  EXPECT_EQ(Vmv.Class, InstClass::V_TO_SCALAR);
  EXPECT_EQ(Vmv.Srcs, (Regs{v(4)}));
}

TEST(InstDecodeTest, UnknownOpcodeHasNoOperands) {
  Inst Unknown = decode(0x0000000b, "custom0");
  EXPECT_EQ(Unknown.Class, InstClass::UNKNOWN);
  EXPECT_TRUE(Unknown.Srcs.empty());
}

TEST(InstDecodeTest, LeavesDestinationsToSpike) {
  Inst I;
  I.Encoding = 0x00b50533; // add a0, a0, a1
  I.Disasm = "add     a0, a0, a1";
  I.Dsts = {x(10)};
  decodeInst(I);
  EXPECT_EQ(I.Dsts, (Regs{x(10)}));
}

} // namespace
