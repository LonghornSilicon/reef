//===- VectorTest.cpp - Vector module tests -------------------------------===//

#include "vector/Vector.h"
#include "common/Inst.h"
#include "vector/VectorPipeline.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

using namespace reef_perf;

/// \returns an instruction \p Mnemonic of class \p Class at LMUL x 8 =
/// \p LMUL8.
static Inst getVectorInst(const char *Mnemonic, uint8_t LMUL8,
                          InstClass Class = InstClass::V_ALU) {
  Inst I;
  I.Mnemonic = Mnemonic;
  I.Class = Class;
  I.LMUL8 = LMUL8;
  return I;
}

// Pool indices of getM3Config().
static constexpr size_t AluPool = 0;
static constexpr size_t PermuteReducePool = 5;

/// \returns M3's organization, as configs/machines/m3.yaml lists it.
static VectorPipelineConfig getM3Config() {
  VectorPipelineConfig Config;
  Config.StationEntries = {8, 8, 4, 8, 4};
  Config.Pools = {{"alu", 2, 0},      {"mul", 2, 1}, {"div", 1, 2},
                  {"fdiv", 1, 2},     {"fma", 2, 3}, {"permute_reduce", 1, 4},
                  {"fp_reduce", 1, 4}};
  return Config;
}

/// \returns a vector add of \p Uops uops.
static VectorOp getAdd(uint32_t Uops) {
  VectorOp Op;
  Op.Pool = AluPool;
  Op.Uops = Uops;
  Op.Latency = 2;
  return Op;
}

/// \returns a load of \p Uops registers, booked in a fake LSU that takes each
/// register as soon as asked and keeps it 5 cycles, as M3's does between
/// vector loads. \p Earliests records the cycle each register was offered.
static VectorOp getLoad(uint32_t Uops, std::vector<Cycle> &Earliests) {
  VectorOp Op;
  Op.Uops = Uops;
  Op.BookLsu = [&Earliests, FreeAt = Cycle(0)](uint32_t,
                                               Cycle Earliest) mutable {
    Earliests.push_back(Earliest);
    VectorLsuResponse Response;
    Response.Start = std::max(Earliest, FreeAt);
    Response.Done = Response.Start + 3;
    FreeAt = Response.Start + 5;
    return Response;
  };
  return Op;
}

namespace {

TEST(VectorUopsTest, OneUopPerRegisterOfTheGroup) {
  EXPECT_EQ(getVectorUops(getVectorInst("vadd.vv", 8)), 1U);
  EXPECT_EQ(getVectorUops(getVectorInst("vadd.vv", 32)), 4U);
  EXPECT_EQ(getVectorUops(getVectorInst("vadd.vv", 4)), 1U); // LMUL = 1/2
}

TEST(VectorUopsTest, WideningAndNarrowingUseTheDoubleGroup) {
  EXPECT_EQ(getVectorUops(getVectorInst("vwmul.vv", 32, InstClass::V_MUL)), 8U);
  EXPECT_EQ(getVectorUops(getVectorInst("vnsrl.wi", 16)), 4U);
  EXPECT_EQ(getVectorUops(getVectorInst("vwadd.vv", 4)), 1U);
  // Not narrowing, despite the "vn".
  EXPECT_EQ(getVectorUops(getVectorInst("vnmsac.vv", 8, InstClass::V_MUL)), 1U);
}

TEST(VectorUopsTest, ReductionWalksItsSourceGroup) {
  EXPECT_EQ(getVectorUops(getVectorInst("vwredsum.vs", 64, InstClass::V_PERM)),
            8U);
}

TEST(VectorUopsTest, ScalarAndMaskRegisterOpsTakeOneUop) {
  for (const char *Op : {"vmv.s.x", "vmv.x.s", "vcpop.m", "vmand.mm"})
    EXPECT_EQ(getVectorUops(getVectorInst(Op, 64)), 1U) << Op;
  EXPECT_EQ(getVectorUops(getVectorInst("vmv4r.v", 8)), 4U);
}

TEST(VectorGroupTest, EveryInstructionLandsInOneGroup) {
  struct Case {
    const char *Mnemonic;
    InstClass Class;
    VectorGroup Group;
  };
  for (const Case &C : std::initializer_list<Case>{
           {"vadd.vv", InstClass::V_ALU, VectorGroup::AluAddSub},
           {"vwsubu.wx", InstClass::V_ALU, VectorGroup::AluAddSub},
           {"vand.vv", InstClass::V_ALU, VectorGroup::Alu},
           {"vmv.v.x", InstClass::V_ALU, VectorGroup::Alu},
           {"vmv.x.s", InstClass::V_TO_SCALAR, VectorGroup::Alu},
           {"vfmv.v.f", InstClass::V_FP, VectorGroup::Alu},
           {"vmseq.vv", InstClass::V_ALU, VectorGroup::Compare},
           {"vmadc.vvm", InstClass::V_ALU, VectorGroup::Compare},
           {"vwmul.vv", InstClass::V_MUL, VectorGroup::Mul},
           {"vnmsac.vv", InstClass::V_MUL, VectorGroup::Mul},
           {"vrem.vx", InstClass::V_DIV, VectorGroup::Div},
           {"vfsqrt.v", InstClass::V_FDIV, VectorGroup::FDiv},
           {"vfmacc.vv", InstClass::V_FP, VectorGroup::Fp},
           {"vmflt.vf", InstClass::V_FP, VectorGroup::Fp},
           {"vslidedown.vi", InstClass::V_PERM, VectorGroup::Permute},
           {"vcompress.vm", InstClass::V_PERM, VectorGroup::Permute},
           {"vredsum.vs", InstClass::V_PERM, VectorGroup::Reduce},
           {"vcpop.m", InstClass::V_TO_SCALAR, VectorGroup::Reduce},
           {"viota.m", InstClass::V_ALU, VectorGroup::Reduce},
           {"vwredsum.vs", InstClass::V_PERM, VectorGroup::WideningReduce},
           {"vfredusum.vs", InstClass::V_PERM, VectorGroup::FpReduce},
       })
    EXPECT_EQ(getVectorGroup(getVectorInst(C.Mnemonic, 8, C.Class)), C.Group)
        << C.Mnemonic;
}

TEST(VectorLatencyTest, OnlyAddSubtractFamilyUsesTheSecondAluStage) {
  Inst I;
  for (const char *Op : {"vadd.vv", "vwsubu.wx", "vmslt.vi", "vmaxu.vv"}) {
    I.Mnemonic = Op;
    EXPECT_TRUE(usesAddSubStage(I)) << Op;
  }
  for (const char *Op : {"vand.vv", "vsll.vi", "vmv.s.x", "vmv.x.s"}) {
    I.Mnemonic = Op;
    EXPECT_FALSE(usesAddSubStage(I)) << Op;
  }
}

TEST(InOrderSlotsTest, FillsACycleThenMovesOn) {
  InOrderSlots Slots(2);
  EXPECT_EQ(Slots.take(5), 5U);
  EXPECT_EQ(Slots.take(5), 5U);
  EXPECT_EQ(Slots.take(5), 6U);
  EXPECT_EQ(Slots.take(3), 6U); // Never earlier than the previous item.
  EXPECT_EQ(Slots.take(9), 9U);
}

TEST(QueueOccupancyTest, EntryFreesTheCycleAfterItLeaves) {
  QueueOccupancy Queue(2);
  EXPECT_EQ(Queue.getFreeCycle(), 0U);
  Queue.recordLeave(10);
  Queue.recordLeave(12);
  EXPECT_EQ(Queue.getFreeCycle(), 11U); // Waits for the item 2 ahead.
  Queue.recordLeave(15);
  EXPECT_EQ(Queue.getFreeCycle(), 13U);
}

// The vector add of the M3 waveform: scalar dispatch in D = 72, DE1 in 74,
// the ALU in 77, vector ROB retire in 80 = D + 8.
TEST(VectorPipelineTest, AddRetiresEightCyclesAfterDispatch) {
  VectorPipeline Pipeline(getM3Config());
  VectorTiming T = Pipeline.schedule(72, getAdd(1), {});
  EXPECT_EQ(T.CommandQueuePop, 74U);
  EXPECT_EQ(T.FirstStart, 77U);
  EXPECT_EQ(T.Retire, 80U);
}

TEST(VectorPipelineTest, DependentDispatchesWhenItsProducerRetires) {
  VectorPipeline Pipeline(getM3Config());
  VectorTiming Producer = Pipeline.schedule(0, getAdd(1), {});
  // Dispatch passes the first cycle the consumer may start: retire + 1.
  VectorTiming Consumer =
      Pipeline.schedule(1, getAdd(1), {Producer.Retire + 1});
  EXPECT_EQ(Consumer.FirstStart, Producer.Retire + 1);
  EXPECT_EQ(Consumer.Retire, Producer.Retire + 4);
}

TEST(VectorPipelineTest, FullVectorRobHoldsBackDispatch) {
  VectorPipeline Pipeline(getM3Config());
  // 8 uops fill the vector ROB; the first of them retires in cycle 8.
  VectorTiming Group = Pipeline.schedule(0, getAdd(8), {});
  EXPECT_EQ(Group.Retire, 11U);
  // The next uop dispatches in 8, reusing that entry in the cycle it frees.
  VectorTiming Next = Pipeline.schedule(0, getAdd(1), {});
  EXPECT_EQ(Next.FirstStart, 9U);
  EXPECT_EQ(Next.Retire, 12U);
}

// vmseq at LMUL=4 (vmseq_m4_indep): its 4 uops all run on the first ALU.
TEST(VectorPipelineTest, CompareRunsOnTheFirstAluOnly) {
  VectorPipeline Pipeline(getM3Config());
  VectorOp Compare = getAdd(4);
  Compare.FirstUnitOnly = true;
  Compare.OneRobEntry = true;
  VectorTiming T = Pipeline.schedule(0, Compare, {});
  EXPECT_EQ(T.FirstStart, 5U);
  EXPECT_EQ(T.Retire, 5U + 3 + 2 + 1); // The fourth uop starts in 8.
}

// The vwredsum of gemv_int8's waveform: 8 uops, one every 2 cycles, through
// the single permute/reduction unit; its 4-entry station holds back the rest.
TEST(VectorPipelineTest, WideningReductionTakesTwoCyclesPerUop) {
  VectorPipeline Pipeline(getM3Config());
  VectorOp Reduce;
  Reduce.Pool = PermuteReducePool;
  Reduce.Uops = 8;
  Reduce.Latency = 4;
  Reduce.Occupancy = 2;
  Reduce.OneRobEntry = true;
  VectorTiming T = Pipeline.schedule(0, Reduce, {});
  EXPECT_EQ(T.FirstStart, 5U);
  EXPECT_EQ(T.Retire, 5U + 7 * 2 + 4 + 1);
  // A younger add waits behind the reduction's uops in dispatch.
  VectorTiming Add = Pipeline.schedule(0, getAdd(1), {});
  EXPECT_GT(Add.FirstStart, 5U + 3 * 2);
}

TEST(VectorPipelineTest, OneRobEntryLeavesRoomForMoreUops) {
  VectorPipeline Pipeline(getM3Config());
  VectorOp Reduce;
  Reduce.Pool = PermuteReducePool;
  Reduce.Uops = 4;
  Reduce.Latency = 3;
  Reduce.OneRobEntry = true;
  Pipeline.schedule(0, Reduce, {});
  // 7 ROB entries remain, so 7 adds dispatch without waiting for a retire.
  VectorTiming Adds = Pipeline.schedule(0, getAdd(7), {});
  EXPECT_LT(Adds.FirstStart, 9U);
}

// Two pools behind one 1-entry station: the second instruction waits for the
// station although its own pool is idle.
TEST(VectorPipelineTest, PoolsSharingAStationWaitForItsEntries) {
  VectorPipelineConfig Config;
  Config.StationEntries = {1};
  Config.Pools = {{"slow", 1, 0}, {"fast", 1, 0}};
  VectorPipeline Pipeline(Config);
  VectorOp Slow;
  Slow.Pool = 0;
  Slow.Latency = 10;
  Slow.Occupancy = 10;
  VectorOp Fast;
  Fast.Pool = 1;
  VectorTiming First = Pipeline.schedule(0, Slow, {});
  VectorTiming Second = Pipeline.schedule(0, Fast, {});
  EXPECT_EQ(First.FirstStart, 5U);
  // The station entry frees when Slow starts; Fast dispatches the cycle after
  // and starts the cycle after that.
  EXPECT_EQ(Second.FirstStart, 7U);
}

// The first vle32 of vec_add_m1's waveform: scalar dispatch in D = 71, the LSU
// takes it in 76 = D + 5 and the vector ROB retires it in 81.
TEST(VectorPipelineTest, LoadReachesTheLsuFiveCyclesAfterDispatch) {
  VectorPipeline Pipeline(getM3Config());
  std::vector<Cycle> Earliests;
  VectorTiming T = Pipeline.schedule(71, getLoad(1, Earliests), {});
  EXPECT_EQ(Earliests, std::vector<Cycle>{76});
  EXPECT_EQ(T.FirstStart, 76U);
  EXPECT_EQ(T.Retire, 81U);
}

TEST(VectorPipelineTest, LoadRegistersGoToTheLsuInOrder) {
  VectorPipeline Pipeline(getM3Config());
  std::vector<Cycle> Earliests;
  VectorTiming T = Pipeline.schedule(0, getLoad(2, Earliests), {});
  ASSERT_EQ(Earliests.size(), 2U);
  EXPECT_EQ(T.FirstStart, 5U);
  EXPECT_EQ(T.Retire, 15U); // The second register starts in 10.
}

// vec_add_m1's vadd dispatches in the cycle its second load retires and
// retires 4 cycles later, as on M3.
TEST(VectorPipelineTest, AddDependingOnALoadDispatchesWhenItRetires) {
  VectorPipeline Pipeline(getM3Config());
  std::vector<Cycle> Earliests;
  VectorTiming Load = Pipeline.schedule(0, getLoad(1, Earliests), {});
  VectorTiming Add = Pipeline.schedule(1, getAdd(1), {Load.Retire + 1});
  EXPECT_EQ(Add.FirstStart, Load.Retire + 1);
  EXPECT_EQ(Add.Retire, Load.Retire + 4);
}

// vec_add_m4: uop i of the vadd waits only for register i of the load, so it
// starts before the load's last register retires.
TEST(VectorPipelineTest, EachUopWaitsOnlyForTheRegisterItReads) {
  VectorPipeline Pipeline(getM3Config());
  std::vector<Cycle> Earliests;
  VectorTiming Load = Pipeline.schedule(0, getLoad(4, Earliests), {});
  ASSERT_EQ(Load.UopRetires.size(), 4U);
  std::vector<Cycle> Ready;
  for (Cycle Retire : Load.UopRetires)
    Ready.push_back(Retire + 1);
  VectorTiming Add = Pipeline.schedule(1, getAdd(4), Ready);
  EXPECT_EQ(Add.FirstStart, Load.UopRetires[0] + 1);
  EXPECT_LT(Add.FirstStart, Load.Retire);
  EXPECT_EQ(Add.Retire, Load.Retire + 4);
}

} // namespace
