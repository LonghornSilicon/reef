//===- Lsu.h - The load/store unit ------------------------------*- C++ -*-===//
///
/// \file
/// The load/store unit: the queue and the single slot every scalar, FP and
/// vector memory instruction goes through (one LSU serves both on M3). When
/// the slot is free, the LSU hands the instruction's accesses (the addresses
/// Spike recorded) to the memory module, which says when the transfer starts,
/// how long it holds the slot and its latency.
///
/// Scalar and FP loads and stores come from Dispatch through the queue, which
/// Dispatch sees through credits; an entry frees when its instruction starts.
/// Vector loads and stores go through the vector module's pipeline, which
/// books the slot one 16-byte vector register at a time
/// (accessVectorRegister()), as M3's RVV backend does. Each register pays a
/// fixed overhead, then its own memory request for the lines it touches.
/// The slot holds one instruction at a time and walks its registers; loading
/// the next instruction takes `lsu_vector_switch_cycles` after the last.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_CORE_LSU_H
#define REEF_PERF_CORE_LSU_H

#include "common/Inst.h"
#include "common/Interfaces.h"
#include "common/ResourcePool.h"

#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

#include <cstdint>
#include <vector>

namespace reef_perf {

/// The load/store unit. Tree location: top.core.lsu.
class Lsu : public sparta::Unit, public VectorLsuInterface {
public:
  // Parameter names are config keys, so they keep Sparta's snake_case.
  // NOLINTBEGIN(readability-identifier-naming)
  struct LsuParameterSet : public sparta::ParameterSet {
    explicit LsuParameterSet(sparta::TreeNode *Node)
        : sparta::ParameterSet(Node) {}
    PARAMETER(uint32_t, lsu_queue_entries, 4, "LSU queue entries")
    PARAMETER(uint32_t, lsu_vector_load_reg_overhead, 0,
              "Extra LSU cycles per vector register of a vector load")
    PARAMETER(uint32_t, lsu_vector_store_reg_overhead, 0,
              "Extra LSU cycles per vector register of a vector store")
    PARAMETER(uint32_t, lsu_vector_switch_cycles, 0,
              "Cycles the slot stays busy after a vector instruction's last "
              "register, loading the next instruction")
  };

  /// Sparta resource name; ResourceFactory requires this exact member name.
  static constexpr const char *name = "lsu";
  // NOLINTEND(readability-identifier-naming)

  /// Creates the unit on \p Node with parameters \p Params.
  Lsu(sparta::TreeNode *Node, const LsuParameterSet *Params);

  /// Connects the memory module \p NewMem, which must outlive the unit.
  void setMemory(MemoryInterface *NewMem) { Mem = NewMem; }

  /// Books the slot for one register of a vector load or store, after the
  /// vector module's LSU reservation station issues it.
  VectorLsuResponse
  accessVectorRegister(const VectorLsuRequest &Request) override;

  /// \returns {lsu}, for the end-of-run summary.
  std::vector<const ResourcePool *> getPools() const { return {&Pipe}; }

private:
  void receiveInst(const InstRef &I);
  void sendInitialCredits();

  const uint32_t QueueEntries;
  const uint32_t VectorLoadRegOverhead;
  const uint32_t VectorStoreRegOverhead;
  const uint32_t VectorSwitchCycles;
  MemoryInterface *Mem = nullptr;
  ResourcePool Pipe; // One memory instruction at a time.

  sparta::DataInPort<InstRef> InInstsPort{&unit_port_set_, "in_insts", 1};
  sparta::DataOutPort<uint32_t> OutCreditsPort{&unit_port_set_, "out_credits"};
  sparta::Counter NumExecuted{&unit_stat_set_, "num_executed",
                              "Scalar and FP memory instructions executed",
                              sparta::Counter::COUNT_NORMAL};
  sparta::Counter NumVectorRegisters{
      &unit_stat_set_, "num_vector_registers",
      "Vector registers loaded or stored for the vector module",
      sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf

#endif // REEF_PERF_CORE_LSU_H
