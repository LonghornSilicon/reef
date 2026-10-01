// Execute: every execution resource in the core, modelled as "pools".
//
// A pool has `count` identical units. An instruction occupies one unit for
// `occupancy` cycles (1 = fully pipelined) and its result is available
// `latency` cycles after it starts. If all units are busy, the instruction
// waits inside Execute (Dispatch does not see scalar pool contention; the
// LSU and vector pools have queues that Dispatch does see, via credits).
//
// SUPER-COARSE BEHAVIOUR (deliberately simpler than the M3 RTL):
//   * One flat latency per class; no data-dependent divider latency.
//   * The LSU is one pool: cost = (distinct 16-byte lines touched) x
//     lsu_cycles_per_line. No separate vector/scalar paths.
//   * The vector unit is one pool of `vec_units` identical lanes. An
//     instruction occupies a lane for (uops x cycles_per_uop), where
//     uops = ceil(vl * SEW / VLEN). No per-type functional units, no
//     decode width, no vector ROB.
//
// Result timing convention used by the scoreboard:
//   result_ready_cycle = start + latency - 1
// i.e. with latency 1 a dependent instruction can dispatch in the cycle the
// producer starts executing (back-to-back issue, as on CoralNPU).

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Inst.hpp"
#include "sparta/events/StartupEvent.hpp"
#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

namespace coralnpu_perf {

class Execute : public sparta::Unit {
 public:
  class ExecuteParameterSet : public sparta::ParameterSet {
   public:
    explicit ExecuteParameterSet(sparta::TreeNode* n) : sparta::ParameterSet(n) {}
    // Scalar integer
    PARAMETER(uint32_t, alu_count, 4, "ALUs (also run branches, jumps, CSRs, system ops)")
    PARAMETER(uint32_t, alu_latency, 1, "ALU latency")
    PARAMETER(uint32_t, mul_count, 1, "Multipliers")
    PARAMETER(uint32_t, mul_latency, 2, "Multiply latency")
    PARAMETER(uint32_t, mul_occupancy, 1, "Cycles a multiply blocks the multiplier")
    PARAMETER(uint32_t, div_latency, 32, "Divide latency (flat; no early-out)")
    PARAMETER(uint32_t, div_occupancy, 32, "Cycles a divide blocks the divider")
    // Scalar FP
    PARAMETER(uint32_t, fpu_latency, 3, "FP add/mul/fma/convert latency")
    PARAMETER(uint32_t, fpu_occupancy, 1, "Cycles an FP op blocks the FPU")
    PARAMETER(uint32_t, fdiv_latency, 12, "FP divide/sqrt latency")
    PARAMETER(uint32_t, fdiv_occupancy, 12, "Cycles an FP divide/sqrt blocks the divider")
    // Load/store unit
    PARAMETER(uint32_t, lsu_queue_entries, 4, "LSU queue entries (seen by dispatch)")
    PARAMETER(uint32_t, lsu_latency, 2, "Load-to-use latency for a single-line access")
    PARAMETER(uint32_t, lsu_line_bytes, 16, "Bytes per memory transaction")
    PARAMETER(uint32_t, lsu_cycles_per_line, 1, "Cycles the LSU is busy per line transaction")
    // Vector unit
    PARAMETER(uint32_t, vec_queue_entries, 8, "Vector command queue entries (seen by dispatch)")
    PARAMETER(uint32_t, vec_units, 2, "Identical vector execution lanes")
    PARAMETER(uint32_t, vlen_bits, 128, "Vector register length (VLEN)")
    PARAMETER(uint32_t, vec_latency, 4, "Latency of a one-uop vector op")
    PARAMETER(uint32_t, vec_cycles_per_uop, 1, "Cycles a lane is busy per uop")
    PARAMETER(uint32_t, vec_div_cycles_per_uop, 8, "Cycles per uop for vector divide/sqrt")
    PARAMETER(uint32_t, vec_to_scalar_latency, 2,
              "Extra latency for vector results written to scalar/FP registers")
  };

  static const char* name;

  Execute(sparta::TreeNode* node, const ExecuteParameterSet* p);

  struct PoolStats {
    std::string name;
    uint32_t count;
    uint64_t ops;
    uint64_t busy_cycles;  // summed over all units in the pool
  };
  std::vector<PoolStats> poolStats() const;

  // A group of identical execution units.
  struct Pool {
    std::string name;
    std::vector<uint64_t> free_at;  // per unit: first cycle it is free again
    uint64_t ops = 0;
    uint64_t busy_cycles = 0;

    // Books the earliest-free unit for `occupancy` cycles, starting no earlier
    // than `earliest`. Returns the start cycle.
    uint64_t reserve(uint64_t earliest, uint64_t occupancy);
  };

 private:
  void receiveInst_(const InstPtr& inst);
  void sendInitialCredits_();
  uint32_t distinctLines_(const Inst& inst) const;
  uint32_t vectorUops_(const Inst& inst) const;

  const ExecuteParameterSet* p_;
  Pool alu_, mul_, div_, fpu_, fdiv_, lsu_, vec_;

  sparta::DataInPort<InstPtr> in_insts_{&unit_port_set_, "in_insts", 1};
  sparta::DataOutPort<uint32_t> out_lsu_credits_{&unit_port_set_, "out_lsu_credits"};
  sparta::DataOutPort<uint32_t> out_vec_credits_{&unit_port_set_, "out_vec_credits"};

  sparta::Counter num_executed_{&unit_stat_set_, "num_executed",
                                "Instructions executed", sparta::Counter::COUNT_NORMAL};
  sparta::Counter num_unknown_{&unit_stat_set_, "num_unknown_class",
                               "Instructions of unknown class (timed as ALU)",
                               sparta::Counter::COUNT_NORMAL};
};

}  // namespace coralnpu_perf
