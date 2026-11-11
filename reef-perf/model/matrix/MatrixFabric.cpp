//===- MatrixFabric.cpp - Compute-fabric timing of a matrix MAC -----------===//

#include "matrix/MatrixFabric.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

using namespace reef_perf;

Dataflow reef_perf::parseDataflow(const std::string &Name) {
  if (Name == "output")
    return Dataflow::Output;
  if (Name == "weight")
    return Dataflow::Weight;
  if (Name == "input")
    return Dataflow::Input;
  throw std::invalid_argument("mtx_dataflow: unknown dataflow '" + Name +
                              "' (output, weight, input)");
}

Delivery reef_perf::parseDelivery(const std::string &Name,
                                  const std::string &Param) {
  if (Name == "broadcast")
    return Delivery::Broadcast;
  if (Name == "forwarded")
    return Delivery::Forwarded;
  throw std::invalid_argument(Param + ": unknown value '" + Name +
                              "' (broadcast, forwarded)");
}

Mapping reef_perf::parseMapping(const std::string &Name) {
  if (Name == "coral")
    return Mapping::Coral;
  if (Name == "generic")
    return Mapping::Generic;
  throw std::invalid_argument("mtx_mapping: unknown value '" + Name +
                              "' (coral, generic)");
}

Reduction reef_perf::parseReduction(const std::string &Name) {
  if (Name == "in_place")
    return Reduction::InPlace;
  if (Name == "chain")
    return Reduction::Chain;
  if (Name == "tree")
    return Reduction::Tree;
  throw std::invalid_argument("mtx_reduction: unknown value '" + Name +
                              "' (in_place, chain, tree)");
}

static uint64_t ceilDiv(uint64_t A, uint64_t B) { return (A + B - 1) / B; }

void reef_perf::checkFabricConfig(const FabricConfig &Config) {
  auto AtLeastOne = [](uint32_t Value, const char *Name) {
    if (Value == 0)
      throw std::invalid_argument(std::string(Name) + " must be at least 1");
  };
  AtLeastOne(Config.Rows, "mtx_pe_rows");
  AtLeastOne(Config.Cols, "mtx_pe_cols");
  AtLeastOne(Config.MacsPerPeInt8, "mtx_macs_per_pe_int8");
  AtLeastOne(Config.MacsPerPeBf16, "mtx_macs_per_pe_bf16");
  AtLeastOne(Config.MacsPerPeFp32, "mtx_macs_per_pe_fp32");
  AtLeastOne(Config.ClockDivider, "mtx_fabric_clock_divider");
  // The Coral rules (dataflow, PE count) are checked by one full-tile MAC.
  getMacCost(Config, 16, 16, 4, MatrixType::Int8);
}

/// \returns the levels of a binary adder tree over \p Inputs inputs.
static uint64_t getTreeLevels(uint64_t Inputs) {
  uint64_t Levels = 0;
  while ((uint64_t{1} << Levels) < Inputs)
    ++Levels;
  return Levels;
}

static uint32_t getMacsPerPe(const FabricConfig &Config, MatrixType Type) {
  switch (Type) {
  case MatrixType::Int8:
    return Config.MacsPerPeInt8;
  case MatrixType::Bf16:
    return Config.MacsPerPeBf16;
  case MatrixType::Fp32:
    return Config.MacsPerPeFp32;
  }
  return 0;
}

FabricCost reef_perf::getMacCost(const FabricConfig &Config, uint32_t TM,
                                 uint32_t TN, uint32_t TK, MatrixType Type) {
  uint64_t Lanes = getMacsPerPe(Config, Type);
  if (Config.Rows == 0 || Config.Cols == 0 || Lanes == 0 ||
      Config.ClockDivider == 0)
    throw std::invalid_argument("matrix fabric needs rows, columns, MAC lanes "
                                "for the input type and a clock divider");
  // A zero dimension still costs one step; the RTL rejects tm or tn = 0.
  uint64_t M = std::max<uint32_t>(TM, 1);
  uint64_t N = std::max<uint32_t>(TN, 1);
  uint64_t K = std::max<uint32_t>(TK, 1);
  uint64_t KSteps = ceilDiv(K, Lanes); // K is spread over a PE's lanes.

  // Which tile dimensions map to rows and columns, and which streams in time.
  uint64_t SpatialRows = M;
  uint64_t SpatialCols = N;
  uint64_t Temporal = KSteps;
  if (Config.Flow == Dataflow::Weight) {
    SpatialRows = KSteps;
    SpatialCols = N;
    Temporal = M;
  } else if (Config.Flow == Dataflow::Input) {
    SpatialRows = KSteps;
    SpatialCols = M;
    Temporal = N;
  }
  uint64_t UsedRows = std::min<uint64_t>(Config.Rows, SpatialRows);
  uint64_t UsedCols = std::min<uint64_t>(Config.Cols, SpatialCols);

  FabricCost Cost;
  if (Config.Map == Mapping::Coral) {
    if (Config.Flow != Dataflow::Output)
      throw std::invalid_argument(
          "mtx_mapping coral models CoralNPU's output-stationary array; use "
          "mtx_mapping generic with mtx_dataflow weight or input");
    // CoralNPU sizes its array as NUM_PE = TE * TE * COMPRATIO and walks a
    // tile in PROCESS_DELAY = 1 / COMPRATIO steps (rvv_backend_define.svh:
    // 210-220). Each step covers TE / 2 / PROCESS_DELAY rows of each half;
    // rows 8-15 run in the upper blocks in parallel, and tn does not change
    // the count (cntMax00, zvt_pe_array.sv:216-217). 64 PEs: 4 steps of 2
    // rows, as at 8a5e1b31.
    constexpr uint64_t TileEdge = 16;
    uint64_t NumPe = uint64_t{Config.Rows} * Config.Cols;
    uint64_t Steps = TileEdge * TileEdge / NumPe;
    if (Steps * NumPe != TileEdge * TileEdge || Steps > TileEdge / 2 ||
        (Steps & (Steps - 1)) != 0)
      throw std::invalid_argument(
          "mtx_mapping coral needs mtx_pe_rows x mtx_pe_cols of 32, 64, 128 "
          "or 256 PEs (a power-of-two share of a 16 x 16 tile)");
    uint64_t RowsPerStep = TileEdge / 2 / Steps;
    Cost.Passes = ceilDiv(std::min<uint64_t>(M, TileEdge / 2), RowsPerStep);
  } else {
    Cost.Passes =
        ceilDiv(SpatialRows, Config.Rows) * ceilDiv(SpatialCols, Config.Cols);
  }
  Cost.Busy = Cost.Passes * Temporal;
  if (Config.Flow != Dataflow::Output) {
    uint64_t Load = Config.StationaryLoadCycles != 0
                        ? Config.StationaryLoadCycles
                        : UsedRows;
    // Double buffering hides every load but the first.
    Cost.Busy += Config.DoubleBufferStationary ? Load : Cost.Passes * Load;
  }

  uint64_t Fill = 0;
  if (Config.RowOperand == Delivery::Forwarded)
    Fill += (UsedCols - 1) * Config.HopLatency;
  if (Config.ColOperand == Delivery::Forwarded)
    Fill += (UsedRows - 1) * Config.HopLatency;
  uint64_t Drain = 0;
  if (Config.Reduce == Reduction::Chain)
    Drain = (UsedRows - 1) * Config.HopLatency;
  else if (Config.Reduce == Reduction::Tree)
    Drain = getTreeLevels(UsedRows * Lanes) * Config.TreeLevelLatency;

  uint64_t Latency = Fill + Cost.Busy + Drain + Config.PipelineDepth +
                     Config.AccumulateLatency;
  Cost.UsefulMacs = M * N * K;
  Cost.CapacityMacs = uint64_t{Config.Rows} * Config.Cols * Lanes * Cost.Busy;
  Cost.Occupancy = Config.PipelineAcrossOps ? Cost.Busy : Latency;
  Cost.Busy *= Config.ClockDivider;
  Cost.Latency = Latency * Config.ClockDivider;
  Cost.Occupancy *= Config.ClockDivider;
  return Cost;
}
