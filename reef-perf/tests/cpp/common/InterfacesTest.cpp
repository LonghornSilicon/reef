//===- InterfacesTest.cpp - Routing tests ---------------------------------===//

#include "common/Interfaces.h"
#include "common/Inst.h"

#include <gtest/gtest.h>

#include <string>

using namespace reef_perf;

/// \returns the unit an instruction of class \p Class is sent to.
static ExecTarget getTargetOf(InstClass Class) {
  Inst I;
  I.Class = Class;
  return getExecTarget(I);
}

namespace {

TEST(ExecTargetTest, ScalarClassesGoToScalarUnits) {
  for (InstClass Class :
       {InstClass::ALU, InstClass::BRANCH, InstClass::JUMP, InstClass::MUL,
        InstClass::DIV, InstClass::CSR, InstClass::FENCE, InstClass::SYSTEM,
        InstClass::FP, InstClass::FP_DIV, InstClass::UNKNOWN})
    EXPECT_EQ(getTargetOf(Class), ExecTarget::Scalar) << getClassName(Class);
}

TEST(ExecTargetTest, EveryMemoryClassGoesToTheLSU) {
  for (InstClass Class :
       {InstClass::LOAD, InstClass::STORE, InstClass::FP_LOAD,
        InstClass::FP_STORE, InstClass::V_LOAD, InstClass::V_STORE})
    EXPECT_EQ(getTargetOf(Class), ExecTarget::LSU) << getClassName(Class);
}

TEST(ExecTargetTest, VectorArithmeticGoesToTheVectorModule) {
  for (InstClass Class : {InstClass::VSET, InstClass::V_ALU, InstClass::V_MUL,
                          InstClass::V_DIV, InstClass::V_FP, InstClass::V_FDIV,
                          InstClass::V_PERM, InstClass::V_TO_SCALAR})
    EXPECT_EQ(getTargetOf(Class), ExecTarget::Vector) << getClassName(Class);
}

TEST(ExecTargetTest, MatrixGoesToTheMatrixModule) {
  EXPECT_EQ(getTargetOf(InstClass::MATRIX), ExecTarget::Matrix);
  EXPECT_EQ(std::string(getClassName(InstClass::MATRIX)), "matrix");
}

} // namespace
