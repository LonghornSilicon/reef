#include "reef_perf/common/inst.hpp"
#include "reef_perf/common/interfaces.hpp"

#include <gtest/gtest.h>

#include <string>

namespace reef_perf {

namespace {

/** The unit an instruction of one class is sent to.
 *
 *  @param cls Instruction class.
 *  @return Its execution target.
 */
ExecTarget target_of(InstClass cls) {
    Inst inst;
    inst.cls = cls;
    return exec_target(inst);
}

} // namespace

TEST(ExecTargetTest, ScalarClassesGoToScalarUnits) {
    for (const InstClass cls :
         {InstClass::ALU, InstClass::BRANCH, InstClass::JUMP, InstClass::MUL,
          InstClass::DIV, InstClass::CSR, InstClass::FENCE, InstClass::SYSTEM,
          InstClass::FP, InstClass::FP_DIV, InstClass::UNKNOWN}) {
        EXPECT_EQ(target_of(cls), ExecTarget::SCALAR) << class_name(cls);
    }
}

TEST(ExecTargetTest, EveryMemoryClassGoesToTheLsu) {
    for (const InstClass cls :
         {InstClass::LOAD, InstClass::STORE, InstClass::FP_LOAD,
          InstClass::FP_STORE, InstClass::V_LOAD, InstClass::V_STORE}) {
        EXPECT_EQ(target_of(cls), ExecTarget::LSU) << class_name(cls);
    }
}

TEST(ExecTargetTest, VectorArithmeticGoesToTheVectorModule) {
    for (const InstClass cls :
         {InstClass::VSET, InstClass::V_ALU, InstClass::V_MUL, InstClass::V_DIV,
          InstClass::V_FP, InstClass::V_FDIV, InstClass::V_PERM,
          InstClass::V_TO_SCALAR}) {
        EXPECT_EQ(target_of(cls), ExecTarget::VECTOR) << class_name(cls);
    }
}

TEST(ExecTargetTest, MatrixGoesToTheMatrixModule) {
    EXPECT_EQ(target_of(InstClass::MATRIX), ExecTarget::MATRIX);
    EXPECT_EQ(std::string(class_name(InstClass::MATRIX)), "matrix");
}

} // namespace reef_perf
