//===- Vector.h - The vector module -----------------------------*- C++ -*-===//
///
/// \file
/// The vector module: the RVV backend. Executes every instruction Dispatch
/// routes to ExecTarget::Vector (vector arithmetic, permutes, vector-to-scalar
/// moves, vset*). Vector loads and stores go to the core's LSU directly; on
/// M3 they also pass through this pipeline (not modeled yet).
///
/// Each instruction goes through M3's in-order backend (VectorPipeline): the
/// frontend register stage, the command queue, two decode stages, the uop
/// queue, dispatch, a unit, the result FIFO and the vector ROB. It completes,
/// for the core's Rob, when its last uop retires from the vector ROB.
///
/// Simpler than the M3 RTL:
/// - one pool of `vec_lanes` identical units, with a latency per class
///   instead of M3's per-type units and reservation stations (B15);
/// - a dependent uop dispatches when its producer's last uop retires;
/// - vset* stays in the frontend: `vset_latency` cycles, no uops;
/// - Dispatch sees the command queue through credits; an entry frees when DE1
///   takes its instruction.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_VECTOR_VECTOR_H
#define REEF_PERF_VECTOR_VECTOR_H

#include "common/Inst.h"
#include "common/ResourcePool.h"
#include "vector/VectorPipeline.h"

#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

#include <cstdint>
#include <vector>

namespace reef_perf {

/// \returns the uops \p I splits into on a machine with \p VLENBits-bit
/// vector registers: ceil(vl x SEW / VLEN), at least 1.
uint32_t getVectorUops(const Inst &I, uint32_t VLENBits);

/// \returns true if \p I goes through M3's two-stage add/subtract ALU path:
/// add, subtract, carry, saturating and averaging add/subtract, min/max and
/// integer compares (rvv_backend_alu_unit_addsub.sv).
bool usesAddSubStage(const Inst &I);

/// The vector module. Tree location: top.vector.
class Vector : public sparta::Unit {
public:
  // Parameter names are config keys, so they keep Sparta's snake_case.
  // NOLINTBEGIN(readability-identifier-naming)
  struct VectorParameterSet : public sparta::ParameterSet {
    explicit VectorParameterSet(sparta::TreeNode *Node)
        : sparta::ParameterSet(Node) {}
    PARAMETER(uint32_t, vec_queue_entries, 8, "Vector command queue entries")
    PARAMETER(uint32_t, vec_de1_width, 2, "Instructions DE1 takes per cycle")
    PARAMETER(uint32_t, vec_legal_queue_entries, 8,
              "Legal command queue entries")
    PARAMETER(uint32_t, vec_de2_width, 2, "Instructions DE2 takes per cycle")
    PARAMETER(uint32_t, vec_de2_uop_width, 6, "Uops DE2 writes per cycle")
    PARAMETER(uint32_t, vec_uop_queue_entries, 16, "Uop queue entries")
    PARAMETER(uint32_t, vec_dispatch_width, 3, "Uops dispatched per cycle")
    PARAMETER(uint32_t, vec_station_entries, 8, "Reservation station entries")
    PARAMETER(uint32_t, vec_rob_entries, 8, "Vector ROB entries (uops)")
    PARAMETER(uint32_t, vec_retire_width, 4, "Uops retired per cycle")
    PARAMETER(uint32_t, vec_lanes, 2, "Vector execution units")
    PARAMETER(uint32_t, vlen_bits, 128, "VLEN in bits")
    PARAMETER(uint32_t, vec_addsub_latency, 2,
              "Cycles in a unit: add, subtract, min/max and compare")
    PARAMETER(uint32_t, vec_alu_latency, 1,
              "Cycles in a unit: other ALU ops, moves to x/f registers")
    PARAMETER(uint32_t, vec_mul_latency, 2, "Cycles in a unit: multiply")
    PARAMETER(uint32_t, vec_perm_latency, 3,
              "Cycles in a unit: permute and reduction")
    PARAMETER(uint32_t, vec_fp_latency, 4, "Cycles in a unit: FP")
    PARAMETER(uint32_t, vec_cycles_per_uop, 1, "Cycles a unit is busy per uop")
    PARAMETER(uint32_t, vec_div_cycles_per_uop, 8,
              "Cycles per uop for vector divide/sqrt, busy and latency")
    PARAMETER(uint32_t, vec_to_scalar_latency, 0,
              "Cycles from vector ROB retire to a scalar use of the result")
    PARAMETER(uint32_t, vset_latency, 1, "vset* latency")
  };

  /// Sparta resource name; ResourceFactory requires this exact member name.
  static constexpr const char *name = "vector";
  // NOLINTEND(readability-identifier-naming)

  /// Public port: vector instructions in from the core (InstRef).
  static constexpr const char *InInsts = "ports.in_insts";
  /// Public port: command-queue credits out to the core.
  static constexpr const char *OutCredits = "ports.out_credits";

  /// Creates the unit on \p Node with parameters \p Params.
  Vector(sparta::TreeNode *Node, const VectorParameterSet *Params);

  /// \returns {vector}, for the end-of-run summary.
  std::vector<const ResourcePool *> getPools() const { return {&Lanes}; }

private:
  void receiveInst(const InstRef &I);
  void sendInitialCredits();
  /// \returns the cycles a uop of \p I spends in its unit.
  uint64_t getLatency(const Inst &I) const;

  const uint32_t QueueEntries;
  const uint32_t VLENBits;
  const uint32_t AddSubLatency;
  const uint32_t AluLatency;
  const uint32_t MulLatency;
  const uint32_t PermLatency;
  const uint32_t FpLatency;
  const uint32_t CyclesPerUop;
  const uint32_t DivCyclesPerUop;
  const uint32_t ToScalarLatency;
  const uint32_t VsetLatency;
  ResourcePool Lanes;
  VectorPipeline Pipeline;

  sparta::DataInPort<InstRef> InInstsPort{&unit_port_set_, "in_insts", 1};
  sparta::DataOutPort<uint32_t> OutCreditsPort{&unit_port_set_, "out_credits"};
  sparta::Counter NumExecuted{&unit_stat_set_, "num_executed",
                              "Vector instructions executed",
                              sparta::Counter::COUNT_NORMAL};
  sparta::Counter NumUops{&unit_stat_set_, "num_uops",
                          "Vector uops retired from the vector ROB",
                          sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf

#endif // REEF_PERF_VECTOR_VECTOR_H
