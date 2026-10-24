//===- VectorTest.cpp - Vector module tests -------------------------------===//

#include "vector/Vector.h"
#include "common/Inst.h"
#include "common/ResourcePool.h"
#include "vector/VectorPipeline.h"

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
  VectorPipeline Pipeline((VectorPipelineConfig()));
  ResourcePool Units("vector", 2);
  VectorTiming T = Pipeline.schedule(72, 1, 2, 1, 0, Units);
  EXPECT_EQ(T.CommandQueuePop, 74U);
  EXPECT_EQ(T.FirstStart, 77U);
  EXPECT_EQ(T.Retire, 80U);
}

TEST(VectorPipelineTest, DependentDispatchesWhenItsProducerRetires) {
  VectorPipeline Pipeline((VectorPipelineConfig()));
  ResourcePool Units("vector", 2);
  VectorTiming Producer = Pipeline.schedule(0, 1, 2, 1, 0, Units);
  // Dispatch passes the first cycle the consumer may start: retire + 1.
  VectorTiming Consumer =
      Pipeline.schedule(1, 1, 2, 1, Producer.Retire + 1, Units);
  EXPECT_EQ(Consumer.FirstStart, Producer.Retire + 1);
  EXPECT_EQ(Consumer.Retire, Producer.Retire + 4);
}

TEST(VectorPipelineTest, FullVectorRobHoldsBackDispatch) {
  VectorPipeline Pipeline((VectorPipelineConfig()));
  ResourcePool Units("vector", 2);
  // 8 uops fill the vector ROB; the first of them retires in cycle 8.
  VectorTiming Group = Pipeline.schedule(0, 8, 2, 1, 0, Units);
  EXPECT_EQ(Group.Retire, 11U);
  // The next uop dispatches in 9, the cycle after that entry frees.
  VectorTiming Next = Pipeline.schedule(0, 1, 2, 1, 0, Units);
  EXPECT_EQ(Next.FirstStart, 10U);
  EXPECT_EQ(Next.Retire, 13U);
}

} // namespace
