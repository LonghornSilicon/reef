//===- VectorTest.cpp - Vector module tests -------------------------------===//

#include "vector/Vector.h"
#include "common/Inst.h"

#include <gtest/gtest.h>

#include <cstdint>

using namespace reef_perf;

/// \returns a vector ALU instruction with vector length \p VL and element
/// width \p SEWBytes.
static Inst getVectorInst(uint32_t VL, uint8_t SEWBytes) {
  Inst I;
  I.Class = InstClass::V_ALU;
  I.VL = VL;
  I.SEWBytes = SEWBytes;
  return I;
}

namespace {

TEST(VectorUopsTest, OneRegisterIsOneUop) {
  EXPECT_EQ(getVectorUops(getVectorInst(4, 4), 128), 1U);
  EXPECT_EQ(getVectorUops(getVectorInst(16, 1), 128), 1U);
}

TEST(VectorUopsTest, RegisterGroupsSplitIntoOneUopPerRegister) {
  EXPECT_EQ(getVectorUops(getVectorInst(16, 4), 128), 4U); // LMUL = 4
  EXPECT_EQ(getVectorUops(getVectorInst(5, 4), 128), 2U);  // Partly filled.
}

TEST(VectorUopsTest, EmptyInstructionStillTakesOneUop) {
  EXPECT_EQ(getVectorUops(getVectorInst(0, 4), 128), 1U);
}

} // namespace
