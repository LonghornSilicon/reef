#include "reef_perf/common/inst.hpp"
#include "reef_perf/vector/vxu.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace reef_perf {

namespace {

/** A vector instruction with the given vl and element width.
 *
 *  @param vl Vector length.
 *  @param sew_bytes Element width in bytes.
 *  @return The instruction.
 */
Inst vector_inst(std::uint32_t vl, std::uint8_t sew_bytes) {
    Inst inst;
    inst.cls = InstClass::V_ALU;
    inst.vl = vl;
    inst.sew_bytes = sew_bytes;
    return inst;
}

} // namespace

TEST(VectorUopsTest, OneRegisterIsOneUop) {
    EXPECT_EQ(vector_uops(vector_inst(4, 4), 128), 1U);
    EXPECT_EQ(vector_uops(vector_inst(16, 1), 128), 1U);
}

TEST(VectorUopsTest, RegisterGroupsSplitIntoOneUopPerRegister) {
    EXPECT_EQ(vector_uops(vector_inst(16, 4), 128), 4U); // LMUL = 4
    EXPECT_EQ(vector_uops(vector_inst(5, 4), 128), 2U);  // partly filled
}

TEST(VectorUopsTest, EmptyInstructionStillTakesOneUop) {
    EXPECT_EQ(vector_uops(vector_inst(0, 4), 128), 1U);
}

} // namespace reef_perf
