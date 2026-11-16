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
    PARAMETER(uint32_t, lsu_scalar_latency_overhead, 0,
              "Extra scalar LSU cycles added to memory latency")
    PARAMETER(uint32_t, lsu_scalar_occupancy_overhead, 0,
              "Extra cycles a scalar memory instruction holds the LSU")
    PARAMETER(uint32_t, lsu_vector_load_reg_overhead, 0,
              "Extra LSU cycles per vector register of a vector load")
    PARAMETER(uint32_t, lsu_vector_store_reg_overhead, 0,
              "Extra LSU cycles per vector register of a vector store")
    PARAMETER(uint32_t, lsu_vector_switch_cycles, 0,
              "Cycles the slot stays busy after a vector instruction's last "
              "register, loading the next instruction")
    // Zvt tile beats (VectorLsuRequest::IsTile, sent only by the matrix
    // module). The newer CoralNPU 8a5e1b31 LSU, the only one with Zvt,
    // charges a tile load or store 2 cycles per instruction + 1 per 16-byte
    // beat, much less than M3's LSU charges a vector register
    // (docs/lsu-tile-beats.md). With lsu_tile_cost off (the default, and in
    // m3.yaml and m3_vme.yaml), a tile beat costs exactly what a vector
    // register does and the four values below are ignored, so the LSU stays
    // M3's. The matrix_diff experiment turns it on to match the VME RTL.
    PARAMETER(bool, lsu_tile_cost, false,
              "Charge Zvt tile beats their own cost (lsu_tile_*) instead of "
              "the vector-register cost")
    PARAMETER(uint32_t, lsu_tile_inst_overhead, 0,
              "Extra LSU cycles on the first beat of a tile load or store")
    PARAMETER(uint32_t, lsu_tile_beat_overhead, 0,
              "Extra LSU cycles on every beat of a tile load or store")
    PARAMETER(uint32_t, lsu_tile_switch_cycles, 0,
              "Cycles the slot stays busy after a tile instruction's last "
              "beat")
    // CoralNPU's LsuV3 (after M3; in 8a5e1b31). With lsu_v3 off, the
    // default and M3, none of the values below is read. On, it replaces
    // the lsu_scalar_* and lsu_vector_* costs above: the slot takes a whole
    // instruction and walks its 16-byte registers one bus cycle each, and a
    // load that needs nothing from the vector backend starts as soon as the
    // core dispatches it (docs/pinned.md, "LsuV3"). Tile beats keep
    // lsu_tile_* when lsu_tile_cost is on.
    PARAMETER(bool, lsu_v3, false, "Model CoralNPU's LsuV3 instead of M3's LSU")
    PARAMETER(bool, lsu_v3_skip_handshake, true,
              "With lsu_v3: an unmasked unit-stride or strided vector load "
              "starts from the core's dispatch, without waiting for the "
              "vector backend (CoralNPU 3ca02cc6). Off, it waits like a "
              "store")
    PARAMETER(uint32_t, lsu_v3_setup_cycles, 1,
              "Cycles between an instruction reaching an idle LSU and its "
              "first bus cycle")
    PARAMETER(uint32_t, lsu_v3_return_cycles, 2,
              "Cycles from a register's last bus cycle to the LSU handing a "
              "loaded value back, or a vector store's register being done")
    PARAMETER(uint32_t, lsu_v3_scalar_store_cycles, 1,
              "Cycles from a scalar store's bus cycle to its completion")
    PARAMETER(uint32_t, lsu_v3_backend_lead, 0,
              "Cycles before a load uop could issue that the vector backend "
              "can already take its loaded register")
    PARAMETER(bool, lsu_v3_tile, false,
              "With lsu_v3: Zvt tile beats run through the LsuV3 slot too, "
              "in place of lsu_tile_*. A tile load is an unmasked "
              "unit-stride load; a tile store takes its data from the "
              "engine's FIFO and is done lsu_v3_scalar_store_cycles after "
              "its last bus cycle")
    PARAMETER(uint32_t, lsu_v3_handshake_cycles, 1,
              "Cycles from the vector backend handing over store data, an "
              "index or a mask to the register's first bus cycle")
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
  void receiveInstV3(const InstRef &I);
  VectorLsuResponse accessVectorRegisterV3(const VectorLsuRequest &Request);
  void sendInitialCredits();

  const uint32_t QueueEntries;
  const uint32_t ScalarLatencyOverhead;
  const uint32_t ScalarOccupancyOverhead;
  const uint32_t VectorLoadRegOverhead;
  const uint32_t VectorStoreRegOverhead;
  const uint32_t VectorSwitchCycles;
  const bool TileCost;
  const uint32_t TileInstOverhead;
  const uint32_t TileBeatOverhead;
  const uint32_t TileSwitchCycles;
  const bool V3;
  const bool V3SkipHandshake;
  const bool V3Tile;
  const uint32_t V3SetupCycles;
  const uint32_t V3ReturnCycles;
  const uint32_t V3ScalarStoreCycles;
  const uint32_t V3HandshakeCycles;
  const uint32_t V3BackendLead;
  Cycle V3NextBus =
      0; // First bus cycle the instruction's next register can use.
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
