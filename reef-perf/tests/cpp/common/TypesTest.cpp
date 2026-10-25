//===- TypesTest.cpp - Purpose-defined type tests -------------------------===//

#include "common/Types.h"

#include <gtest/gtest.h>

#include <sstream>

using namespace reef_perf;

namespace {

TEST(TypesTest, DenseIndexSeparatesRegisterFiles) {
  EXPECT_EQ(RegId::x(0).getDense(), 0U);
  EXPECT_EQ(RegId::x(31).getDense(), 31U);
  EXPECT_EQ(RegId::f(0).getDense(), 32U);
  EXPECT_EQ(RegId::v(31).getDense(), RegId::NumDense - 1);
}

TEST(TypesTest, SameIndexInDifferentFilesDiffers) {
  EXPECT_NE(RegId::x(5), RegId::f(5));
  EXPECT_EQ(RegId::v(5), RegId(RegFile::V, 5));
  EXPECT_LT(RegId::x(31), RegId::f(0));
}

TEST(TypesTest, PrintsAssemblyName) {
  std::ostringstream OS;
  OS << RegId::x(10) << ' ' << RegId::f(2) << ' ' << RegId::v(4);
  EXPECT_EQ(OS.str(), "x10 f2 v4");
}

} // namespace
