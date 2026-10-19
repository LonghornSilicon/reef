//===- Matrix.h - The matrix module (stub) ----------------------*- C++ -*-===//
///
/// \file
/// The matrix module: a stub with the vector module's interface (instructions
/// in, command-queue credits out). It executes every instruction Dispatch
/// routes to ExecTarget::Matrix. No matrix ISA exists yet, so nothing decodes
/// to InstClass::MATRIX and the module sees no traffic; it is here so the
/// core-matrix interface can be built and tested first.
///
/// Stub behavior: `mtx_units` engines, each busy `mtx_occupancy` cycles per
/// instruction, with a flat `mtx_latency`. All values are placeholders.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_MATRIX_MATRIX_H
#define REEF_PERF_MATRIX_MATRIX_H

#include "common/Inst.h"
#include "common/ResourcePool.h"

#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

#include <cstdint>
#include <vector>

namespace reef_perf {

/// The matrix module. Tree location: top.matrix.
class Matrix : public sparta::Unit {
public:
  // Parameter names are config keys, so they keep Sparta's snake_case.
  // NOLINTBEGIN(readability-identifier-naming)
  class MatrixParameterSet : public sparta::ParameterSet {
  public:
    explicit MatrixParameterSet(sparta::TreeNode *Node)
        : sparta::ParameterSet(Node) {}
    PARAMETER(uint32_t, mtx_queue_entries, 4, "Matrix command queue entries")
    PARAMETER(uint32_t, mtx_units, 1, "Matrix engines")
    PARAMETER(uint32_t, mtx_latency, 8, "Matrix op latency")
    PARAMETER(uint32_t, mtx_occupancy, 4,
              "Cycles a matrix op blocks its engine")
  };

  /// Sparta resource name; ResourceFactory requires this exact member name.
  static constexpr const char *name = "matrix";
  // NOLINTEND(readability-identifier-naming)

  /// Public port: matrix instructions in from the core (InstRef).
  static constexpr const char *InInsts = "ports.in_insts";
  /// Public port: command-queue credits out to the core.
  static constexpr const char *OutCredits = "ports.out_credits";

  /// Creates the unit on \p Node with parameters \p Params.
  Matrix(sparta::TreeNode *Node, const MatrixParameterSet *Params);

  /// \returns {matrix}, for the end-of-run summary.
  std::vector<const ResourcePool *> getPools() const { return {&Engines}; }

private:
  void receiveInst(const InstRef &I);
  void sendInitialCredits();

  const uint32_t QueueEntries;
  const uint32_t Latency;
  const uint32_t Occupancy;
  ResourcePool Engines;

  sparta::DataInPort<InstRef> InInstsPort{&unit_port_set_, "in_insts", 1};
  sparta::DataOutPort<uint32_t> OutCreditsPort{&unit_port_set_, "out_credits"};
  sparta::Counter NumExecuted{&unit_stat_set_, "num_executed",
                              "Matrix instructions executed",
                              sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf

#endif // REEF_PERF_MATRIX_MATRIX_H
