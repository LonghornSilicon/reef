//===- Vector.h - The vector module -----------------------------*- C++ -*-===//
///
/// \file
/// The vector module: the RVV backend's command queue and lanes. Executes
/// every instruction Dispatch routes to ExecTarget::Vector (vector arithmetic,
/// permutes, vector-to-scalar moves, vset*). Vector loads and stores go to the
/// core's LSU, as on M3.
///
/// Super-coarse, deliberately simpler than the M3 RTL:
/// - one pool of `vec_lanes` identical lanes; an instruction holds a lane for
///   uops x vec_cycles_per_uop cycles, where uops = ceil(vl x SEW / VLEN);
/// - no per-type functional units, decode width, vector ROB or chaining;
/// - vset* takes `vset_latency` cycles and uses no lane;
/// - Dispatch sees the command queue through credits; an entry frees when its
///   instruction starts.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_VECTOR_VECTOR_H
#define REEF_PERF_VECTOR_VECTOR_H

#include "common/Inst.h"
#include "common/ResourcePool.h"

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

/// The vector module. Tree location: top.vector.
class Vector : public sparta::Unit {
public:
  // Parameter names are config keys, so they keep Sparta's snake_case.
  // NOLINTBEGIN(readability-identifier-naming)
  struct VectorParameterSet : public sparta::ParameterSet {
    explicit VectorParameterSet(sparta::TreeNode *Node)
        : sparta::ParameterSet(Node) {}
    PARAMETER(uint32_t, vec_queue_entries, 8, "Vector command queue entries")
    PARAMETER(uint32_t, vec_lanes, 2, "Vector execution lanes")
    PARAMETER(uint32_t, vlen_bits, 128, "VLEN in bits")
    PARAMETER(uint32_t, vec_latency, 4, "One-uop vector latency")
    PARAMETER(uint32_t, vec_cycles_per_uop, 1, "Cycles a lane is busy per uop")
    PARAMETER(uint32_t, vec_div_cycles_per_uop, 8,
              "Cycles per uop for vector divide/sqrt")
    PARAMETER(uint32_t, vec_to_scalar_latency, 2,
              "Extra latency for vector results written to x/f regs")
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

  const uint32_t QueueEntries;
  const uint32_t VLENBits;
  const uint32_t Latency;
  const uint32_t CyclesPerUop;
  const uint32_t DivCyclesPerUop;
  const uint32_t ToScalarLatency;
  const uint32_t VsetLatency;
  ResourcePool Lanes;

  sparta::DataInPort<InstRef> InInstsPort{&unit_port_set_, "in_insts", 1};
  sparta::DataOutPort<uint32_t> OutCreditsPort{&unit_port_set_, "out_credits"};
  sparta::Counter NumExecuted{&unit_stat_set_, "num_executed",
                              "Vector instructions executed",
                              sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf

#endif // REEF_PERF_VECTOR_VECTOR_H
