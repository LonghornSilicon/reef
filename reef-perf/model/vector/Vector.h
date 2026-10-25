//===- Vector.h - The vector module -----------------------------*- C++ -*-===//
///
/// \file
/// The vector module: the RVV backend. Executes every instruction Dispatch
/// routes to ExecTarget::Vector: vector arithmetic, permutes, vector-to-scalar
/// moves, vset*, and vector loads and stores.
///
/// Each instruction goes through M3's in-order backend (VectorPipeline): the
/// frontend register stage, the command queue, two decode stages, the uop
/// queue, dispatch, a reservation station and functional unit pool, the
/// result FIFO and the vector ROB. It completes, for the core's Rob, when its
/// last uop retires from the vector ROB. A load or store has one uop per
/// vector register; each waits in the LSU reservation station, then books the
/// core's LSU (VectorLsuInterface), which M3's scalar core shares.
///
/// The functional units are configuration: `vec_stations` and `vec_pools`
/// list the reservation stations and pools, and `vec_<group>_pool` assigns
/// each instruction group (VectorGroup) to one pool, so it also lands in that
/// pool's station. The defaults are M3's per-type organization;
/// configs/machines/m3_unified_vector.yaml is a two-lane alternative.
/// Uop counts, vector ROB entries and latencies stay M3's in every
/// organization.
///
/// Simpler than the M3 RTL:
/// - fixed cycles per uop; M3's integer divider takes longer for wider
///   dividends and reuses its last result for repeated operands;
/// - a dependent uop dispatches when its producer's last uop retires;
/// - vset* stays in the frontend: `vset_latency` cycles, no uops;
/// - Dispatch sees the command queue through credits; an entry frees when DE1
///   takes its instruction.
///
/// `vec_completion_matches_pc` reproduces an M3 RTL bug: the retirement
/// buffer matches vector completions by PC, so in a loop shorter than the
/// buffer, a later iteration retires when an earlier one completes
/// (rtl_bugs.md at the top of the Reef repository). Set it to false for the
/// fixed behavior.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_VECTOR_VECTOR_H
#define REEF_PERF_VECTOR_VECTOR_H

#include "common/Inst.h"
#include "common/Interfaces.h"
#include "common/ResourcePool.h"
#include "vector/VectorPipeline.h"

#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace reef_perf {

/// \returns the uops M3 splits \p I into: one per register of its largest
/// register group (EMUL), whatever its vector length
/// (rvv_backend_decode_unit_ari.sv).
uint32_t getVectorUops(const Inst &I);

/// Sets of vector instructions with the same timing, which the configuration
/// assigns to functional unit pools. Every vector instruction except vset*,
/// loads and stores is in exactly one.
enum class VectorGroup : uint8_t {
  AluAddSub,      ///< Add, subtract, min/max: M3's two-stage ALU path.
  Alu,            ///< Other ALU ops: logic, shifts, moves, mask logic.
  Compare,        ///< Integer compares, vmadc, vmsbc.
  Mul,            ///< Integer multiply and multiply-accumulate.
  Div,            ///< Integer divide and remainder.
  FDiv,           ///< FP divide and square root.
  Fp,             ///< Other FP arithmetic, compares and conversions.
  Permute,        ///< Slides, gathers, compress.
  Reduce,         ///< Integer reductions, vcpop, viota.
  WideningReduce, ///< vwredsum, vwredsumu.
  FpReduce,       ///< FP reductions.
};

/// The number of VectorGroup values.
inline constexpr size_t NumVectorGroups = 11;

/// \returns the instruction group of \p I.
VectorGroup getVectorGroup(const Inst &I);

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
    PARAMETER(uint32_t, vec_rob_entries, 8, "Vector ROB entries (uops)")
    PARAMETER(uint32_t, vec_retire_width, 4, "Uops retired per cycle")
    // How functional units are organized: reservation stations, pools fed by
    // them, and the pool each instruction group runs on.
    PARAMETER(std::vector<std::string>, vec_stations,
              (std::vector<std::string>{"alu", "mul", "div", "fma",
                                        "permute_reduce"}),
              "Reservation station names")
    PARAMETER(std::vector<uint32_t>, vec_station_entries,
              (std::vector<uint32_t>{8, 8, 4, 8, 4}),
              "Entries of each reservation station, in vec_stations order")
    PARAMETER(std::vector<std::string>, vec_pools,
              (std::vector<std::string>{"alu", "mul", "div", "fdiv", "fma",
                                        "permute_reduce", "fp_reduce"}),
              "Functional unit pool names")
    PARAMETER(std::vector<uint32_t>, vec_pool_counts,
              (std::vector<uint32_t>{2, 2, 1, 1, 2, 1, 1}),
              "Functional units in each pool, in vec_pools order")
    PARAMETER(std::vector<std::string>, vec_pool_stations,
              (std::vector<std::string>{"alu", "mul", "div", "div", "fma",
                                        "permute_reduce", "permute_reduce"}),
              "Station feeding each pool, in vec_pools order")
    PARAMETER(uint32_t, vec_lsu_station_entries, 4,
              "Entries of the reservation station for loads and stores")
    PARAMETER(std::string, vec_alu_addsub_pool, "alu",
              "Pool for add, subtract, min/max")
    PARAMETER(std::string, vec_alu_pool, "alu", "Pool for other ALU ops")
    PARAMETER(std::string, vec_compare_pool, "alu", "Pool for integer compares")
    PARAMETER(bool, vec_compare_first_unit_only, true,
              "Only the first unit of its pool runs a compare")
    PARAMETER(std::string, vec_mul_pool, "mul", "Pool for integer multiply")
    PARAMETER(std::string, vec_div_pool, "div", "Pool for integer divide")
    PARAMETER(std::string, vec_fdiv_pool, "fdiv",
              "Pool for FP divide and square root")
    PARAMETER(std::string, vec_fp_pool, "fma", "Pool for other FP ops")
    PARAMETER(std::string, vec_permute_pool, "permute_reduce",
              "Pool for slides, gathers, compress")
    PARAMETER(std::string, vec_reduce_pool, "permute_reduce",
              "Pool for integer reductions, vcpop, viota")
    PARAMETER(std::string, vec_widening_reduce_pool, "permute_reduce",
              "Pool for widening reductions")
    PARAMETER(std::string, vec_fp_reduce_pool, "fp_reduce",
              "Pool for FP reductions")
    PARAMETER(uint32_t, vec_addsub_latency, 2,
              "Cycles in the ALU: add, subtract, min/max and compare")
    PARAMETER(uint32_t, vec_alu_latency, 1,
              "Cycles in the ALU: other ops, moves to x/f registers")
    PARAMETER(uint32_t, vec_mul_latency, 2, "Cycles in a multiplier")
    PARAMETER(uint32_t, vec_div_cycles_per_uop, 13,
              "Cycles per uop for integer divide, busy and latency")
    PARAMETER(uint32_t, vec_fdiv_latency, 9,
              "Cycles in the divider: FP divide and square root")
    PARAMETER(uint32_t, vec_fdiv_cycles_per_uop, 8,
              "Cycles the divider is busy per FP divide uop")
    PARAMETER(uint32_t, vec_fp_latency, 4, "Cycles in an FP multiply-add unit")
    PARAMETER(uint32_t, vec_permute_latency, 3,
              "Cycles in the permute/reduction unit: slides, gathers")
    PARAMETER(uint32_t, vec_permute_cycles_per_uop, 5,
              "Cycles it is busy per slide or gather uop")
    PARAMETER(uint32_t, vec_reduce_latency, 3,
              "Cycles in the permute/reduction unit: integer reductions, "
              "vcpop, viota")
    PARAMETER(uint32_t, vec_widening_reduce_cycles_per_uop, 2,
              "Cycles it is busy per widening-reduction uop; each extra "
              "cycle also adds to the latency")
    PARAMETER(uint32_t, vec_fp_reduce_latency, 13,
              "Cycles in the permute/reduction unit: FP reductions")
    PARAMETER(uint32_t, vec_fp_reduce_cycles_per_uop, 12,
              "Cycles it is busy per FP reduction uop")
    PARAMETER(uint32_t, vec_to_scalar_latency, 0,
              "Cycles from vector ROB retire to a scalar use of the result")
    PARAMETER(uint32_t, vset_latency, 1, "vset* latency")
    PARAMETER(bool, vec_completion_matches_pc, true,
              "The core's retirement buffer matches vector results and "
              "store completions to entries by PC, so an older instance of "
              "the same PC completes newer ones (M3's RTL bug)")
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

  /// Connects the core's LSU \p NewLsu, which runs vector loads and stores
  /// and must outlive the unit.
  void setLsu(VectorLsuInterface *NewLsu) { Lsu = NewLsu; }

  /// \returns the functional unit pools, for the end-of-run summary.
  std::vector<const ResourcePool *> getPools() const {
    return Pipeline.getPools();
  }

private:
  void receiveInst(const InstRef &I);
  void sendInitialCredits();
  /// \returns what \p I needs from the backend.
  VectorOp getOp(const Inst &I) const;
  /// \returns per uop of \p I's \p Uops, the first cycle the vector
  /// registers it reads allow it to start.
  std::vector<Cycle> getSourcesReady(const Inst &I, uint32_t Uops) const;
  /// Records when \p I's destination registers are ready, from timing \p T.
  void recordResults(const Inst &I, const VectorTiming &T);
  /// \returns when the core's retirement buffer sees \p I, with timing \p T,
  /// complete.
  Cycle getScalarComplete(const Inst &I, const VectorTiming &T);

  const uint32_t QueueEntries;
  const uint32_t AddSubLatency;
  const uint32_t AluLatency;
  const uint32_t MulLatency;
  const uint32_t DivCyclesPerUop;
  const uint32_t FDivLatency;
  const uint32_t FDivCyclesPerUop;
  const uint32_t FpLatency;
  const uint32_t PermuteLatency;
  const uint32_t PermuteCyclesPerUop;
  const uint32_t ReduceLatency;
  const uint32_t WideningReduceCyclesPerUop;
  const uint32_t FpReduceLatency;
  const uint32_t FpReduceCyclesPerUop;
  const uint32_t ToScalarLatency;
  const uint32_t VsetLatency;
  const bool CompareFirstUnitOnly;
  const bool CompletionMatchesPc;
  std::array<size_t, NumVectorGroups> GroupPools; // Pool of each group.
  VectorPipeline Pipeline;
  VectorLsuInterface *Lsu = nullptr;
  // Per vector register: the first cycle a uop reading it may start, the
  // cycle after the uop writing it retires. Every writer is a vector
  // instruction, and they arrive in program order.
  std::array<Cycle, RegId::NumPerFile> RegReady{};
  // Per PC: the cycles a write for it marks retirement-buffer entries done,
  // from instances that may still be in flight.
  std::unordered_map<Address, std::vector<Cycle>> PcDoneEvents;

  sparta::DataInPort<InstRef> InInstsPort{&unit_port_set_, "in_insts", 1};
  sparta::DataOutPort<uint32_t> OutCreditsPort{&unit_port_set_, "out_credits"};
  sparta::Counter NumExecuted{&unit_stat_set_, "num_executed",
                              "Vector instructions executed",
                              sparta::Counter::COUNT_NORMAL};
  sparta::Counter NumUops{&unit_stat_set_, "num_uops",
                          "Vector uops dispatched to a functional unit",
                          sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf

#endif // REEF_PERF_VECTOR_VECTOR_H
