//===- ScalarExec.h - Scalar functional units -------------------*- C++ -*-===//
///
/// \file
/// Scalar execution: the integer and FP functional units as ResourcePools.
/// Executes everything Dispatch routes to ExecTarget::Scalar.
///
/// Uses configurable fixed latencies, with optional operand-dependent integer
/// divide timing. A busy unit only delays the start; Dispatch does not see it.
/// Scoreboard convention: ResultReadyCycle = start + latency - 1, so with
/// latency 1 a dependent instruction dispatches in the producer's start cycle.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_CORE_SCALAREXEC_H
#define REEF_PERF_CORE_SCALAREXEC_H

#include "common/Inst.h"
#include "common/ResourcePool.h"

#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

#include <cstdint>
#include <vector>

namespace reef_perf {

/// The scalar functional units. Tree location: top.core.scalar_exec.
class ScalarExec : public sparta::Unit {
public:
  // Parameter names are config keys, so they keep Sparta's snake_case.
  // NOLINTBEGIN(readability-identifier-naming)
  struct ScalarExecParameterSet : public sparta::ParameterSet {
    explicit ScalarExecParameterSet(sparta::TreeNode *Node)
        : sparta::ParameterSet(Node) {}
    PARAMETER(uint32_t, alu_count, 4,
              "ALUs (also run branches, jumps, CSRs, system ops)")
    PARAMETER(uint32_t, alu_latency, 1, "ALU latency")
    PARAMETER(uint32_t, mul_count, 1, "Multipliers")
    PARAMETER(uint32_t, mul_latency, 2, "Multiply latency")
    PARAMETER(uint32_t, mul_occupancy, 1,
              "Cycles a multiply blocks the multiplier")
    PARAMETER(uint32_t, div_latency, 32, "Divide latency")
    PARAMETER(uint32_t, div_occupancy, 32, "Cycles a divide blocks the divider")
    PARAMETER(bool, div_data_dependent, false,
              "Use operand-dependent M3 divide timing")
    PARAMETER(uint32_t, div_base, 2,
              "Fixed cycles added to M3's divide iterations")
    PARAMETER(uint32_t, fpu_latency, 3, "FP op latency")
    PARAMETER(uint32_t, fpu_occupancy, 1, "Cycles an FP op blocks the FPU")
    PARAMETER(uint32_t, fdiv_latency, 12, "FP divide/sqrt latency")
    PARAMETER(uint32_t, fdiv_occupancy, 12,
              "Cycles an FP divide/sqrt blocks the divider")
  };

  /// Sparta resource name; ResourceFactory requires this exact member name.
  static constexpr const char *name = "scalar_exec";
  // NOLINTEND(readability-identifier-naming)

  /// Creates the unit on \p Node with parameters \p Params.
  ScalarExec(sparta::TreeNode *Node, const ScalarExecParameterSet *Params);

  /// \returns alu, mul, div, fpu and fdiv, for the end-of-run summary.
  std::vector<const ResourcePool *> getPools() const {
    return {&ALU, &MUL, &DIV, &FPU, &FDIV};
  }

private:
  void receiveInst(const InstRef &I);

  const uint32_t ALULatency;
  const uint32_t MULLatency;
  const uint32_t MULOccupancy;
  const uint32_t DIVLatency;
  const uint32_t DIVOccupancy;
  const bool DIVDataDependent;
  const uint32_t DIVBase;
  const uint32_t FPULatency;
  const uint32_t FPUOccupancy;
  const uint32_t FDIVLatency;
  const uint32_t FDIVOccupancy;
  ResourcePool ALU;
  ResourcePool MUL;
  ResourcePool DIV;
  ResourcePool FPU;
  ResourcePool FDIV;

  sparta::DataInPort<InstRef> InInstsPort{&unit_port_set_, "in_insts", 1};
  sparta::Counter NumExecuted{&unit_stat_set_, "num_executed",
                              "Instructions executed",
                              sparta::Counter::COUNT_NORMAL};
  sparta::Counter NumUnknown{&unit_stat_set_, "num_unknown_class",
                             "Instructions of unknown class (timed as ALU)",
                             sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf

#endif // REEF_PERF_CORE_SCALAREXEC_H
