#include "reef_perf/common/inst.hpp"
#include "reef_perf/frontend/inst_decode.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace reef_perf {

namespace {

using Regs = std::vector<std::uint16_t>;

/** Decodes one instruction with the given vector configuration.
 *
 *  @param encoding Raw 32-bit instruction.
 *  @param disasm Disassembly, as Spike prints it.
 *  @param lmul8 LMUL times eight.
 *  @return The decoded instruction.
 */
Inst decode(std::uint32_t encoding, const std::string& disasm,
            std::uint8_t lmul8 = 8) {
    Inst inst;
    inst.encoding = encoding;
    inst.disasm = disasm;
    inst.lmul8 = lmul8;
    decode_inst(inst);
    return inst;
}

constexpr std::uint16_t x(std::uint16_t reg) { return kXBase + reg; }
constexpr std::uint16_t f(std::uint16_t reg) { return kFBase + reg; }
constexpr std::uint16_t v(std::uint16_t reg) { return kVBase + reg; }

} // namespace

TEST(InstDecodeTest, ExtractsMnemonic) {
    EXPECT_EQ(decode(0x00b50533, "add     a0, a0, a1").mnemonic, "add");
}

TEST(InstDecodeTest, IntegerAluSkipsX0) {
    const Inst li = decode(0x01000513, "li      a0, 16"); // addi a0, zero, 16
    EXPECT_EQ(li.cls, InstClass::ALU);
    EXPECT_EQ(li.srcs, Regs{});
    EXPECT_EQ(li.dsts, (Regs{x(10)}));

    const Inst add = decode(0x00b50533, "add     a0, a0, a1");
    EXPECT_EQ(add.srcs, (Regs{x(10), x(11)}));
    EXPECT_EQ(add.dsts, (Regs{x(10)}));
}

TEST(InstDecodeTest, MultiplyAndDivide) {
    EXPECT_EQ(decode(0x02a585b3, "mul     a1, a1, a0").cls, InstClass::MUL);
    EXPECT_EQ(decode(0x02c55533, "divu    a0, a0, a2").cls, InstClass::DIV);
}

TEST(InstDecodeTest, LoadsAndStores) {
    const Inst load = decode(0x00052583, "lw      a1, 0(a0)");
    EXPECT_EQ(load.cls, InstClass::LOAD);
    EXPECT_EQ(load.srcs, (Regs{x(10)}));
    EXPECT_EQ(load.dsts, (Regs{x(11)}));
    EXPECT_TRUE(load.is_memory());

    const Inst store = decode(0x00c5a023, "sw      a2, 0(a1)");
    EXPECT_EQ(store.cls, InstClass::STORE);
    EXPECT_EQ(store.srcs, (Regs{x(11), x(12)}));
    EXPECT_EQ(store.dsts, Regs{});
}

TEST(InstDecodeTest, CsrReadOfMcycle) {
    const Inst csr = decode(0xb0002d73, "csrr    s10, mcycle");
    EXPECT_EQ(csr.cls, InstClass::CSR);
    EXPECT_EQ(csr.srcs, Regs{});
    EXPECT_EQ(csr.dsts, (Regs{x(26)}));
}

TEST(InstDecodeTest, MpauseIsSystem) {
    EXPECT_EQ(decode(0x08000073, "mpause").cls, InstClass::SYSTEM);
}

TEST(InstDecodeTest, FloatingPointUsesFRegisters) {
    const Inst fadd = decode(0x003170d3, "fadd.s  ft1, ft2, ft3");
    EXPECT_EQ(fadd.cls, InstClass::FP);
    EXPECT_EQ(fadd.srcs, (Regs{f(2), f(3)}));
    EXPECT_EQ(fadd.dsts, (Regs{f(1)}));
}

TEST(InstDecodeTest, VsetvliReadsAvlAndWritesVl) {
    const Inst vset = decode(0x0c0572d7, "vsetvli t0, a0, e8, m1, ta, ma");
    EXPECT_EQ(vset.cls, InstClass::VSET);
    EXPECT_EQ(vset.srcs, (Regs{x(10)}));
    EXPECT_EQ(vset.dsts, (Regs{x(5)}));
    EXPECT_TRUE(vset.is_vector());
}

TEST(InstDecodeTest, VectorUnitStrideLoad) {
    const Inst vle = decode(0x02058087, "vle8.v  v1, (a1)");
    EXPECT_EQ(vle.cls, InstClass::V_LOAD);
    EXPECT_EQ(vle.srcs, (Regs{x(11)}));
    EXPECT_EQ(vle.dsts, (Regs{v(1)}));
}

TEST(InstDecodeTest, VectorAddAtLmulOne) {
    const Inst vadd = decode(0x02108157, "vadd.vv v2, v1, v1");
    EXPECT_EQ(vadd.cls, InstClass::V_ALU);
    EXPECT_EQ(vadd.srcs, (Regs{v(1), v(1)}));
    EXPECT_EQ(vadd.dsts, (Regs{v(2)}));
}

TEST(InstDecodeTest, VectorAddAtLmulFourUsesRegisterGroups) {
    const Inst vadd = decode(0x02440657, "vadd.vv v12, v4, v8", 32);
    EXPECT_EQ(vadd.srcs,
              (Regs{v(8), v(9), v(10), v(11), v(4), v(5), v(6), v(7)}));
    EXPECT_EQ(vadd.dsts, (Regs{v(12), v(13), v(14), v(15)}));
}

TEST(InstDecodeTest, MaskedVectorOpReadsV0) {
    const Inst vadd = decode(0x00108157, "vadd.vv v2, v1, v1, v0.t");
    EXPECT_EQ(vadd.srcs, (Regs{v(1), v(1), v(0)}));
}

TEST(InstDecodeTest, MultiplyAccumulateReadsDestination) {
    const Inst vmacc = decode(0xb620a1d7, "vmacc.vv v3, v1, v2");
    EXPECT_EQ(vmacc.cls, InstClass::V_MUL);
    EXPECT_EQ(vmacc.srcs, (Regs{v(1), v(2), v(3)}));
    EXPECT_EQ(vmacc.dsts, (Regs{v(3)}));
}

TEST(InstDecodeTest, ReductionIsPermuteClass) {
    const Inst vred = decode(0x02302257, "vredsum.vs v4, v3, v0");
    EXPECT_EQ(vred.cls, InstClass::V_PERM);
    EXPECT_EQ(vred.srcs, (Regs{v(0), v(3)}));
    EXPECT_EQ(vred.dsts, (Regs{v(4)}));
}

TEST(InstDecodeTest, VmvXsWritesScalar) {
    const Inst vmv = decode(0x42402657, "vmv.x.s a2, v4");
    EXPECT_EQ(vmv.cls, InstClass::V_TO_SCALAR);
    EXPECT_EQ(vmv.srcs, (Regs{v(4)}));
    EXPECT_EQ(vmv.dsts, (Regs{x(12)}));
}

TEST(InstDecodeTest, UnknownOpcodeHasNoOperands) {
    const Inst unknown = decode(0x0000000b, "custom0");
    EXPECT_EQ(unknown.cls, InstClass::UNKNOWN);
    EXPECT_TRUE(unknown.srcs.empty());
    EXPECT_TRUE(unknown.dsts.empty());
}

} // namespace reef_perf
