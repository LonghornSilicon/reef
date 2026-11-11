//===- MatrixFabric.h - Compute-fabric timing of a matrix MAC ---*- C++ -*-===//
///
/// \file
/// Timing of one tile multiply-accumulate (C[TM x TN] += A^T[TM x TK] B[TK x
/// TN]) on a configurable MAC array. The array is described by independent
/// mechanisms rather than by a named design, so CoralNPU's outer-product
/// engine, weight-, output- and input-stationary systolic arrays and
/// adder-tree arrays are points in one parameter space (docs/matrix.md).
///
/// This is an analytic model. Only the CoralNPU setting has an RTL to check
/// against; every other setting needs a reference (a cycle-level array
/// simulator or an RTL prototype) before its numbers are trusted.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_MATRIX_MATRIXFABRIC_H
#define REEF_PERF_MATRIX_MATRIXFABRIC_H

#include "common/Inst.h"

#include <cstdint>
#include <string>

namespace reef_perf {

/// Which operand stays in the PEs while the others stream past.
enum class Dataflow : uint8_t {
  Output, ///< C stays; K streams in time. Rows map M, columns map N.
  Weight, ///< B stays; M streams in time. Rows map K, columns map N.
  Input   ///< A stays; N streams in time. Rows map K, columns map M.
};

/// How an operand reaches the PEs along one array dimension.
enum class Delivery : uint8_t {
  Broadcast, ///< Every PE in the row/column gets it in the same cycle.
  Forwarded  ///< It moves one PE per hop (systolic), adding skew.
};

/// Where partial sums over the array's rows are added.
enum class Reduction : uint8_t {
  InPlace, ///< Each PE accumulates its own output; nothing moves.
  Chain,   ///< Partial sums move down a column, one PE per hop.
  Tree     ///< An adder tree sums the rows' (and lanes') products.
};

/// How a tile's rows and columns map onto the array's steps.
enum class Mapping : uint8_t {
  /// CoralNPU's PE array: 256 / PEs steps (PROCESS_DELAY) of
  /// 8 / PROCESS_DELAY tile rows per half, whatever tn is (ROWS_PER_CNT and
  /// cntMax00, zvt_pe_array.sv:216-217); at 64 PEs, min(4, ceil(tm / 2)).
  /// Output-stationary only.
  Coral,
  /// Rows and columns tile the array in rows x cols blocks (one pass per
  /// block); for the sweep's other designs. Checked against the cycle-level
  /// reference in experiments/array_ref (docs/matrix.md, D4).
  Generic
};

/// \returns the Mapping named \p Name ("coral", "generic").
/// \throws std::invalid_argument for any other name.
Mapping parseMapping(const std::string &Name);
/// \returns the Dataflow named \p Name ("output", "weight", "input").
/// \throws std::invalid_argument for any other name.
Dataflow parseDataflow(const std::string &Name);
/// \returns the Delivery named \p Name ("broadcast", "forwarded").
/// \throws std::invalid_argument naming \p Param for any other name.
Delivery parseDelivery(const std::string &Name, const std::string &Param);
/// \returns the Reduction named \p Name ("in_place", "chain", "tree").
/// \throws std::invalid_argument for any other name.
Reduction parseReduction(const std::string &Name);

/// One MAC array. Defaults are CoralNPU's engine at 8a5e1b31 where known
/// (docs/matrix.md lists each value's source).
struct FabricConfig {
  uint32_t Rows = 8;                ///< PE rows.
  uint32_t Cols = 8;                ///< PE columns.
  uint32_t MacsPerPeInt8 = 4;       ///< int8 MACs per PE per cycle (K lanes).
  uint32_t MacsPerPeBf16 = 2;       ///< bf16 MACs per PE per cycle.
  uint32_t MacsPerPeFp32 = 1;       ///< fp32 MACs per PE per cycle.
  Dataflow Flow = Dataflow::Output; ///< Which operand stays in the PEs.
  Delivery RowOperand = Delivery::Broadcast; ///< Enters at the left edge.
  Delivery ColOperand = Delivery::Broadcast; ///< Enters at the top edge.
  uint32_t HopLatency = 1;                   ///< Cycles per PE-to-PE hop.
  Reduction Reduce = Reduction::InPlace;     ///< Where partial sums add.
  uint32_t TreeLevelLatency = 1;             ///< Cycles per adder-tree level.
  uint32_t PipelineDepth = 9;     ///< PE pipeline stages after the last step.
  uint32_t AccumulateLatency = 0; ///< Extra read-modify-write of C.
  /// Cycles to load the stationary operand per pass; 0 means one PE row per
  /// cycle (the rows used). Output-stationary arrays load nothing.
  uint32_t StationaryLoadCycles = 0;
  /// The next pass's stationary operand loads during this pass.
  bool DoubleBufferStationary = false;
  /// Fill, drain and pipeline overlap the next operation, so the array is
  /// busy only for the compute steps.
  bool PipelineAcrossOps = true;
  uint32_t ClockDivider = 1;    ///< Core cycles per fabric cycle.
  Mapping Map = Mapping::Coral; ///< How partial tiles map onto the steps.
};

/// Timing of one MAC operation, in core cycles.
struct FabricCost {
  uint64_t Passes = 0;       ///< Array-sized blocks the tile splits into.
  uint64_t Busy = 0;         ///< Cycles of compute steps and loads.
  uint64_t Latency = 0;      ///< Start to accumulator written.
  uint64_t Occupancy = 0;    ///< Cycles the array is unavailable to others.
  uint64_t UsefulMacs = 0;   ///< TM * TN * TK.
  uint64_t CapacityMacs = 0; ///< MACs the array could do in Busy cycles.
};

/// Checks \p Config before a run.
/// \throws std::invalid_argument naming the first bad parameter: a zero row,
/// column, MAC-lane count or clock divider, the Coral mapping with a
/// dataflow other than output-stationary, or a Coral PE count that is not
/// 32, 64, 128 or 256.
void checkFabricConfig(const FabricConfig &Config);

/// \returns the timing of a TM x TN x TK MAC with \p Type inputs on \p Config.
/// \throws std::invalid_argument if the array has no rows, columns or MAC
/// lanes for \p Type, ClockDivider is 0, or the Coral mapping is used with
/// a dataflow other than output-stationary.
FabricCost getMacCost(const FabricConfig &Config, uint32_t TM, uint32_t TN,
                      uint32_t TK, MatrixType Type);

} // namespace reef_perf

#endif // REEF_PERF_MATRIX_MATRIXFABRIC_H
