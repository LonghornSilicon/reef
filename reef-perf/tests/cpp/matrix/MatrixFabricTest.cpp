//===- MatrixFabricTest.cpp - Matrix fabric timing tests ------------------===//

#include "matrix/MatrixFabric.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace reef_perf;

namespace {

TEST(MatrixFabricTest, CoralDefaultsFullTile) {
  // 64 PEs x 4 int8 lanes = 256 MACs/cycle; 1024 MACs take 4 cycles, the
  // PE-busy count measured for vtmmu tm = tn = 16, tk = 4.
  FabricCost C = getMacCost(FabricConfig{}, 16, 16, 4, MatrixType::Int8);
  EXPECT_EQ(C.Passes, 4U);
  EXPECT_EQ(C.Busy, 4U);
  EXPECT_EQ(C.Occupancy, 4U);
  EXPECT_EQ(C.Latency, 4U + 9U); // + PipelineDepth
  EXPECT_EQ(C.UsefulMacs, 1024U);
  EXPECT_EQ(C.CapacityMacs, 1024U);
}

TEST(MatrixFabricTest, CoralDefaultsOtherTypes) {
  EXPECT_EQ(getMacCost(FabricConfig{}, 16, 16, 1, MatrixType::Fp32).Busy, 4U);
  EXPECT_EQ(getMacCost(FabricConfig{}, 16, 16, 2, MatrixType::Bf16).Busy, 4U);
}

TEST(MatrixFabricTest, CoralPartialTileFollowsTheFormula) {
  // The formula gives 2 cycles for tm = 8. CoralNPU's PE array takes 4
  // (zvt_pe_array.sv:216, cntMax00): an open discrepancy (docs/matrix.md).
  EXPECT_EQ(getMacCost(FabricConfig{}, 8, 16, 4, MatrixType::Int8).Busy, 2U);
}

TEST(MatrixFabricTest, WeightStationarySystolic) {
  FabricConfig F;
  F.Rows = 16;
  F.Cols = 16;
  F.MacsPerPeInt8 = 1;
  F.Flow = Dataflow::Weight;
  F.RowOperand = Delivery::Forwarded;
  F.ColOperand = Delivery::Forwarded;
  F.Reduce = Reduction::Chain;
  F.DoubleBufferStationary = true;
  F.PipelineDepth = 1;
  FabricCost C = getMacCost(F, 16, 16, 4, MatrixType::Int8);
  // Rows map K (4 used), columns map N (16), M = 16 streams.
  EXPECT_EQ(C.Passes, 1U);
  EXPECT_EQ(C.Busy, 16U + 4U); // stream + one exposed load of 4 rows
  // Fill (16 - 1) + (4 - 1), drain (4 - 1), pipeline 1.
  EXPECT_EQ(C.Latency, 15U + 3U + 20U + 3U + 1U);
  EXPECT_EQ(C.UsefulMacs, 1024U);
  EXPECT_EQ(C.CapacityMacs, 16U * 16U * 20U);
}

TEST(MatrixFabricTest, StationaryLoadsWithoutDoubleBuffering) {
  FabricConfig F;
  F.Rows = 2;
  F.Cols = 16;
  F.MacsPerPeInt8 = 1;
  F.Flow = Dataflow::Weight;
  F.StationaryLoadCycles = 5;
  FabricCost C = getMacCost(F, 16, 16, 4, MatrixType::Int8);
  EXPECT_EQ(C.Passes, 2U);               // K = 4 over 2 rows
  EXPECT_EQ(C.Busy, 2U * 16U + 2U * 5U); // every pass loads
}

TEST(MatrixFabricTest, AdderTreeDrain) {
  FabricConfig F;
  F.Rows = 8;
  F.Cols = 8;
  F.MacsPerPeInt8 = 4;
  F.Flow = Dataflow::Weight;
  F.Reduce = Reduction::Tree;
  F.TreeLevelLatency = 2;
  F.PipelineDepth = 0;
  F.DoubleBufferStationary = true;
  FabricCost C = getMacCost(F, 16, 16, 4, MatrixType::Int8);
  // K = 4 fits one PE's lanes: 1 row used; tree over 1 x 4 inputs = 2 levels.
  EXPECT_EQ(C.Passes, 2U);
  EXPECT_EQ(C.Busy, 2U * 16U + 1U);
  EXPECT_EQ(C.Latency, C.Busy + 2U * 2U);
}

TEST(MatrixFabricTest, PipelineAcrossOpsAndClockDivider) {
  FabricConfig F;
  F.PipelineAcrossOps = false;
  F.ClockDivider = 2;
  FabricCost C = getMacCost(F, 16, 16, 4, MatrixType::Int8);
  EXPECT_EQ(C.Latency, 2U * 13U);
  EXPECT_EQ(C.Occupancy, C.Latency);
  EXPECT_EQ(C.Busy, 8U);
}

TEST(MatrixFabricTest, RejectsBadConfigs) {
  FabricConfig F;
  F.MacsPerPeBf16 = 0;
  EXPECT_THROW(getMacCost(F, 16, 16, 2, MatrixType::Bf16),
               std::invalid_argument);
  EXPECT_THROW(parseDataflow("row"), std::invalid_argument);
  EXPECT_THROW(parseDelivery("multicast"), std::invalid_argument);
  EXPECT_THROW(parseReduction("ring"), std::invalid_argument);
  EXPECT_EQ(parseDataflow("weight"), Dataflow::Weight);
}

} // namespace
