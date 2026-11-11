//===- MatrixFabricTest.cpp - Matrix fabric timing tests ------------------===//

#include "matrix/MatrixFabric.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

using namespace reef_perf;

/// \returns the message \p F throws as std::invalid_argument, or "".
static std::string getError(const std::function<void()> &F) {
  try {
    F();
  } catch (const std::invalid_argument &E) {
    return E.what();
  }
  return "";
}

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

TEST(MatrixFabricTest, CoralPartialTilesFollowThePeArray) {
  // busy = min(4, ceil(tm / 2)), no tn term (zvt_pe_array.sv:216-217); the
  // VME RTL takes 4 cycles per independent 8x8 vtmmu (mac_indep_tm8tn8).
  auto Busy = [](uint32_t TM, uint32_t TN) {
    return getMacCost(FabricConfig{}, TM, TN, 4, MatrixType::Int8).Busy;
  };
  EXPECT_EQ(Busy(1, 16), 1U);
  EXPECT_EQ(Busy(4, 16), 2U);
  EXPECT_EQ(Busy(8, 8), 4U);
  EXPECT_EQ(Busy(16, 4), 4U);
  EXPECT_EQ(Busy(16, 16), 4U);
}

TEST(MatrixFabricTest, CoralStepsAreTheProcessDelay) {
  FabricConfig F;
  EXPECT_EQ(getCoralSteps(F), 4U); // 64 PEs.
  F.Cols = 16;
  EXPECT_EQ(getCoralSteps(F), 2U); // 128 PEs.
  F.Cols = 12;
  EXPECT_THROW(getCoralSteps(F), std::invalid_argument);
}

TEST(MatrixFabricTest, CoralStepsScaleWithThePeCount) {
  // PROCESS_DELAY = 256 / NUM_PE steps of 8 / PROCESS_DELAY rows per half.
  auto Busy = [](uint32_t Cols, uint32_t TM) {
    FabricConfig F;
    F.Cols = Cols;
    return getMacCost(F, TM, 16, 4, MatrixType::Int8).Busy;
  };
  EXPECT_EQ(Busy(16, 16), 2U); // 128 PEs.
  EXPECT_EQ(Busy(16, 4), 1U);
  EXPECT_EQ(Busy(16, 5), 2U);
  EXPECT_EQ(Busy(32, 16), 1U); // 256 PEs.
  EXPECT_EQ(Busy(4, 16), 8U);  // 32 PEs.
  EXPECT_EQ(Busy(4, 3), 3U);
  FabricConfig Odd;
  Odd.Cols = 12; // 96 PEs: not a power-of-two share of the tile.
  EXPECT_THROW(getMacCost(Odd, 16, 16, 4, MatrixType::Int8),
               std::invalid_argument);
}

TEST(MatrixFabricTest, GenericPartialTileFollowsTheFormula) {
  FabricConfig F;
  F.Map = Mapping::Generic;
  EXPECT_EQ(getMacCost(F, 8, 16, 4, MatrixType::Int8).Busy, 2U);
  EXPECT_EQ(getMacCost(F, 8, 8, 4, MatrixType::Int8).Busy, 1U);
  EXPECT_EQ(getMacCost(F, 16, 16, 4, MatrixType::Int8).Busy, 4U);
}

TEST(MatrixFabricTest, WeightStationarySystolic) {
  FabricConfig F;
  F.Map = Mapping::Generic;
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
  // Fill (16 - 1) along the rows, drain (4 - 1) down the chain, pipeline 1;
  // the weights do not move, so there is no second skew.
  EXPECT_EQ(C.Latency, 15U + 20U + 3U + 1U);
  // Back to back, the next MAC's load hides behind this one's 16 steps.
  EXPECT_EQ(C.Occupancy, 16U);
  EXPECT_EQ(C.UsefulMacs, 1024U);
  EXPECT_EQ(C.CapacityMacs, 16U * 16U * 20U);
}

TEST(MatrixFabricTest, StationaryLoadsWithoutDoubleBuffering) {
  FabricConfig F;
  F.Map = Mapping::Generic;
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
  F.Map = Mapping::Generic;
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
  EXPECT_THROW(parseDelivery("multicast", "mtx_row_operand"),
               std::invalid_argument);
  EXPECT_THROW(parseReduction("ring"), std::invalid_argument);
  EXPECT_THROW(parseMapping("diagonal"), std::invalid_argument);
  FabricConfig Ws;
  Ws.Flow = Dataflow::Weight; // The Coral mapping is output-stationary only.
  EXPECT_THROW(getMacCost(Ws, 16, 16, 4, MatrixType::Int8),
               std::invalid_argument);
  EXPECT_EQ(parseDataflow("weight"), Dataflow::Weight);
}

TEST(MatrixFabricTest, ZeroSizesNameTheParameter) {
  auto Error = [](const std::function<void(FabricConfig &)> &Set) {
    FabricConfig F;
    Set(F);
    return getError([&] { checkFabricConfig(F); });
  };
  EXPECT_EQ(Error([](FabricConfig &F) { F.Rows = 0; }),
            "mtx_pe_rows must be at least 1");
  EXPECT_EQ(Error([](FabricConfig &F) { F.Cols = 0; }),
            "mtx_pe_cols must be at least 1");
  EXPECT_EQ(Error([](FabricConfig &F) { F.MacsPerPeInt8 = 0; }),
            "mtx_macs_per_pe_int8 must be at least 1");
  EXPECT_EQ(Error([](FabricConfig &F) { F.MacsPerPeBf16 = 0; }),
            "mtx_macs_per_pe_bf16 must be at least 1");
  EXPECT_EQ(Error([](FabricConfig &F) { F.MacsPerPeFp32 = 0; }),
            "mtx_macs_per_pe_fp32 must be at least 1");
  EXPECT_EQ(Error([](FabricConfig &F) { F.ClockDivider = 0; }),
            "mtx_fabric_clock_divider must be at least 1");
  EXPECT_NE(Error([](FabricConfig &F) { F.Cols = 12; }).find("mtx_pe_cols"),
            std::string::npos);
  EXPECT_NE(Error([](FabricConfig &F) {
              F.Flow = Dataflow::Weight;
            }).find("mtx_mapping"),
            std::string::npos);
  EXPECT_EQ(Error([](FabricConfig &) {}), "");
}

TEST(MatrixFabricTest, UnknownNamesNameTheParameter) {
  auto Has = [](const std::function<void()> &F, const std::string &Name) {
    return getError(F).find(Name) != std::string::npos;
  };
  EXPECT_TRUE(Has([] { parseDataflow("diagonal"); }, "mtx_dataflow"));
  EXPECT_TRUE(Has([] { parseDelivery("teleport", "mtx_col_operand"); },
                  "mtx_col_operand"));
  EXPECT_TRUE(Has([] { parseReduction("sum"); }, "mtx_reduction"));
  EXPECT_TRUE(Has([] { parseMapping("other"); }, "mtx_mapping"));
}

/// One row of FabricReference.inc: an array, a MAC shape, and the cycle-level
/// reference's latency and back-to-back interval.
struct ReferenceCase {
  uint32_t Rows, Cols, Lanes;
  Dataflow Flow;
  Delivery RowOperand, ColOperand;
  Reduction Reduce;
  uint32_t Hop, TreeLevel, Depth, StationaryLoad;
  bool DoubleBuffer;
  uint32_t M, N, K;
  uint64_t Latency;
  double Interval;
};

const ReferenceCase ReferenceCases[] = {
#include "FabricReference.inc"
};

TEST(MatrixFabricTest, ReferenceArrays) {
  // The generic formulas against experiments/array_ref, a cycle-level
  // schedule of the same arrays (docs/matrix-correlation-log.md, D4). The
  // interval may be a fraction (passes alternate buffers); occupancy is its
  // ceiling.
  for (const ReferenceCase &R : ReferenceCases) {
    FabricConfig F;
    F.Rows = R.Rows;
    F.Cols = R.Cols;
    F.MacsPerPeInt8 = R.Lanes;
    F.Flow = R.Flow;
    F.RowOperand = R.RowOperand;
    F.ColOperand = R.ColOperand;
    F.Reduce = R.Reduce;
    F.HopLatency = R.Hop;
    F.TreeLevelLatency = R.TreeLevel;
    F.PipelineDepth = R.Depth;
    F.StationaryLoadCycles = R.StationaryLoad;
    F.DoubleBufferStationary = R.DoubleBuffer;
    F.Map = Mapping::Generic;
    FabricCost C = getMacCost(F, R.M, R.N, R.K, MatrixType::Int8);
    SCOPED_TRACE(std::to_string(R.Rows) + "x" + std::to_string(R.Cols) +
                 " lanes " + std::to_string(R.Lanes) + " shape " +
                 std::to_string(R.M) + "x" + std::to_string(R.N) + "x" +
                 std::to_string(R.K));
    EXPECT_EQ(C.Latency, R.Latency);
    EXPECT_GE(static_cast<double>(C.Occupancy), R.Interval);
    EXPECT_LT(static_cast<double>(C.Occupancy), R.Interval + 1);
  }
}

} // namespace
