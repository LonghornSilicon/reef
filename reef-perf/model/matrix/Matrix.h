//===- Matrix.h - The matrix module -----------------------------*- C++ -*-===//
///
/// \file
/// The matrix module: a Zvt matrix engine. Executes every instruction
/// Dispatch routes to ExecTarget::Matrix (vtzero, vtdiscard, vtmv.*, vtle*,
/// vtse*, vtmm*; Inst::Mtx says which). mset* are vector configuration and go
/// to the vector module.
///
/// Each instruction goes through:
///  1. the command path: `mtx_cmd_latency` cycles, then its uops issue at
///     `mtx_uops_per_cycle`, in program order, once the vector registers it
///     reads are ready (CoralNPU sends Zvt uops through the RVV backend's
///     in-order dispatch);
///  2. its execution resource, in program order (the next instruction starts
///     once this one releases the controller's head): the MAC array pool
///     (`mtx_num_arrays` arrays, timed by MatrixFabric) for vtmm, the
///     controller for vtzero and moves, and the core's LSU, one 16-byte beat
///     at a time, for tile loads and stores;
///  3. completion: `mtx_complete_latency` cycles after the tile (or vector
///     register, or memory) is written, the core's Rob may retire it.
///
/// A tile scoreboard holds, per architectural tile, the cycle its last write
/// lands; an instruction waits for every tile of its group (1, 2 or 4 tiles
/// by element width). A MAC into a tile whose last writer was a MAC waits
/// only `mtx_acc_raw_cycles` after that MAC started, as CoralNPU's hitRaw
/// check does. Tile contents are not modeled.
///
/// Each default's source is in docs/matrix.md, "Parameters".
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_MATRIX_MATRIX_H
#define REEF_PERF_MATRIX_MATRIX_H

#include "common/Inst.h"
#include "common/Interfaces.h"
#include "common/ResourcePool.h"
#include "matrix/MatrixFabric.h"

#include "sparta/events/PayloadEvent.hpp"
#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace reef_perf {

/// End-of-run matrix statistics, for the JSON summary.
struct MatrixStats {
  /// Instructions executed, by MatrixOp.
  std::array<uint64_t, static_cast<size_t>(MatrixOp::NumOps)> Ops{};
  uint64_t UsefulMacs = 0;      ///< Sum of TM * TN * TK.
  uint64_t CapacityMacs = 0;    ///< MACs the arrays could have done.
  uint64_t ArrayBusyCycles = 0; ///< Fabric busy cycles, summed.
  uint64_t MemBeats = 0;        ///< 16-byte LSU beats for tile loads/stores.
  uint64_t SubtileReads = 0;    ///< 16-byte tile reads (coarse count).
  uint64_t SubtileWrites = 0;   ///< 16-byte tile writes (coarse count).
};

/// The matrix module. Tree location: top.matrix.
class Matrix : public sparta::Unit {
public:
  // Parameter names are config keys, so they keep Sparta's snake_case.
  // NOLINTBEGIN(readability-identifier-naming)
  struct MatrixParameterSet : public sparta::ParameterSet {
    explicit MatrixParameterSet(sparta::TreeNode *Node)
        : sparta::ParameterSet(Node) {}
    // Core interface and command path.
    PARAMETER(uint32_t, mtx_queue_entries, 4, "Matrix command queue entries")
    PARAMETER(uint32_t, mtx_cmd_latency, 4,
              "Cycles from receiving an instruction to issuing its uops")
    PARAMETER(uint32_t, mtx_uops_per_cycle, 3, "Matrix uops issued per cycle")
    PARAMETER(uint32_t, mtx_uop_buffer_entries, 8,
              "Uops the engine's reservation station holds (ZVT_RS_DEPTH)")
    PARAMETER(uint32_t, mtx_complete_latency, 1,
              "Cycles from the result being written to the core seeing the "
              "instruction complete")
    // Compute fabric (MatrixFabric).
    PARAMETER(uint32_t, mtx_num_arrays, 1, "Independent MAC arrays")
    PARAMETER(uint32_t, mtx_pe_rows, 8, "PE rows per array")
    PARAMETER(uint32_t, mtx_pe_cols, 8, "PE columns per array")
    PARAMETER(uint32_t, mtx_macs_per_pe_int8, 4, "int8 MACs per PE per cycle")
    PARAMETER(uint32_t, mtx_macs_per_pe_bf16, 2, "bf16 MACs per PE per cycle")
    PARAMETER(uint32_t, mtx_macs_per_pe_fp32, 1, "fp32 MACs per PE per cycle")
    PARAMETER(std::string, mtx_dataflow, "output",
              "Stationary operand: output, weight or input")
    PARAMETER(std::string, mtx_row_operand, "broadcast",
              "Delivery of the operand entering each row: broadcast or "
              "forwarded")
    PARAMETER(std::string, mtx_col_operand, "broadcast",
              "Delivery of the operand entering each column: broadcast or "
              "forwarded")
    PARAMETER(uint32_t, mtx_hop_latency, 1, "Cycles per PE-to-PE hop")
    PARAMETER(std::string, mtx_mapping, "coral",
              "How partial tiles map onto the array: coral (CoralNPU's PE "
              "array, busy = min(4, ceil(tm/2))) or generic (rows x cols "
              "blocks)")
    PARAMETER(std::string, mtx_reduction, "in_place",
              "Partial-sum reduction: in_place, chain or tree")
    PARAMETER(uint32_t, mtx_tree_level_latency, 1,
              "Cycles per adder-tree level")
    PARAMETER(uint32_t, mtx_pe_pipeline_depth, 9,
              "PE pipeline stages after the last compute step")
    PARAMETER(uint32_t, mtx_accumulate_latency, 0,
              "Extra cycles to read, add and write the accumulator")
    PARAMETER(uint32_t, mtx_acc_raw_cycles, 4,
              "Cycles after a MAC starts until a MAC into the same tile may "
              "start (0: wait until the first MAC's tile write lands)")
    PARAMETER(uint32_t, mtx_stationary_load_cycles, 0,
              "Cycles to load the stationary operand per pass (0: one PE "
              "row per cycle)")
    PARAMETER(bool, mtx_double_buffer_stationary, false,
              "Load the next pass's stationary operand during this pass")
    PARAMETER(bool, mtx_pipeline_across_ops, true,
              "Fill, drain and pipeline overlap the next operation")
    PARAMETER(uint32_t, mtx_fabric_clock_divider, 1,
              "Core cycles per fabric cycle")
    // Tile storage and data movement.
    PARAMETER(uint32_t, mtx_zero_cycles, 0,
              "Cycles vtzero holds the controller; 0 derives it: 256 / PEs "
              "for mtx_mapping coral (PROCESS_DELAY), 4 otherwise")
    PARAMETER(uint32_t, mtx_move_cycles_per_uop, 1,
              "Controller cycles per vtmv uop")
    PARAMETER(uint32_t, mtx_move_latency, 0,
              "Extra cycles, after the cycle following a move's last uop, "
              "until its result can be read")
    PARAMETER(uint32_t, mtx_tile_write_latency, 1,
              "Cycles from a tile load's last data beat to the tile write")
    PARAMETER(uint32_t, mtx_tile_read_latency, 1,
              "Cycles to read a tile slice for a store")
    PARAMETER(uint32_t, mtx_lsu_fifo_entries, 2,
              "Entries in each FIFO between the LSU and the controller "
              "(lsu2vme for tile loads, vme2lsu for tile stores)")
    PARAMETER(bool, mtx_ctrl_waits_for_array_drain, true,
              "vtzero, moves, tile loads and stores wait until no MAC is "
              "in flight")
    // Cost, for weighing designs; 0 leaves the estimate out of the summary.
    PARAMETER(double, mtx_area_per_pe_um2, 0.0, "Area of one PE (um^2)")
    PARAMETER(double, mtx_tile_storage_area_um2, 0.0,
              "Area of the tile storage (um^2)")
    PARAMETER(double, mtx_energy_per_mac_pj, 0.0, "Energy per useful MAC (pJ)")
    PARAMETER(double, mtx_energy_per_subtile_access_pj, 0.0,
              "Energy per 16-byte tile read or write (pJ)")
    PARAMETER(double, mtx_energy_per_mem_beat_pj, 0.0,
              "Energy per 16-byte memory beat (pJ)")
  };

  /// Sparta resource name; ResourceFactory requires this exact member name.
  static constexpr const char *name = "matrix";
  // NOLINTEND(readability-identifier-naming)

  /// Public port: matrix instructions in from the core (InstRef).
  static constexpr const char *InInsts = "ports.in_insts";
  /// Public port: command-queue credits out to the core.
  static constexpr const char *OutCredits = "ports.out_credits";

  /// Creates the unit on \p Node with parameters \p Params.
  /// \throws std::invalid_argument for an unknown dataflow, delivery or
  /// reduction name, or a zero-sized resource.
  Matrix(sparta::TreeNode *Node, const MatrixParameterSet *Params);

  /// Connects the core's LSU \p NewLsu, which runs tile loads and stores and
  /// must outlive the unit.
  void setLsu(VectorLsuInterface *NewLsu) { Lsu = NewLsu; }

  /// Connects the vector module's register scoreboard \p Regs, which must
  /// outlive the unit.
  void setVectorRegisters(VectorRegisterInterface *Regs) { VectorRegs = Regs; }

  /// \returns the command path, the MAC arrays and the controller.
  std::vector<const ResourcePool *> getPools() const {
    return {&Cmd, &Arrays, &Ctrl};
  }

  /// \returns the end-of-run statistics.
  const MatrixStats &getStats() const { return Stats; }

  /// \returns the area estimate in um^2, or a negative value if no area
  /// parameter is set.
  double getAreaUm2() const;

  /// \returns the energy estimate in pJ, or a negative value if no energy
  /// parameter is set.
  double getEnergyPj() const;

private:
  void receiveInst(const InstRef &I);
  void executeInst(const InstRef &I);
  void sendInitialCredits();
  /// \returns the uops CoralNPU splits \p I into.
  static uint32_t getUops(const Inst &I);
  /// \returns the first cycle every vector register \p I reads is ready.
  Cycle getVectorSourcesReady(const Inst &I) const;
  /// \returns the latest entry of \p Ready over \p I's tile group.
  static Cycle getTilesReady(const Inst &I, const std::array<Cycle, 16> &Ready);
  /// Sets \p I's tile group to \p Ready in \p Tiles.
  static void setTilesReady(const Inst &I, std::array<Cycle, 16> &Tiles,
                            Cycle Ready);
  /// Controller and LSU timing of one tile load or store.
  struct Transfer {
    Cycle CtrlFirst = 0; ///< The controller's first beat.
    Cycle CtrlLast = 0;  ///< The controller's last beat.
    Cycle LsuDone = 0;   ///< The LSU finished the last beat (stores).
  };
  /// Runs tile load \p I, its LSU half free to start at \p Now and its
  /// controller half at \p CtrlEarliest, beat by beat through lsu2vme.
  Transfer runTileLoad(const Inst &I, Cycle Now, Cycle CtrlEarliest);
  /// Runs tile store \p I from \p CtrlEarliest, beat by beat through
  /// vme2lsu.
  Transfer runTileStore(const Inst &I, Cycle CtrlEarliest);
  /// \returns the first cycle the FIFO whose recent pops are \p Pops has a
  /// free entry.
  Cycle getFifoFree(const std::deque<Cycle> &Pops) const;
  /// Records a pop at \p Cycle in \p Pops.
  void recordPop(std::deque<Cycle> &Pops, Cycle When) const;
  /// Pushes \p Uops uops into the reservation station in order, from
  /// \p Earliest. \returns the first and last push cycles.
  std::pair<Cycle, Cycle> pushUops(uint32_t Uops, Cycle Earliest);
  /// Sets the pop cycles of the newest \p Uops uops: all at \p Last for a
  /// MAC (\p Together), else spread one per cycle ending at \p Last.
  void popUops(uint32_t Uops, Cycle Last, bool Together);

  const uint32_t QueueEntries;
  const uint32_t CmdLatency;
  const uint32_t UopsPerCycle;
  const uint32_t UopBufferEntries;
  const uint32_t CompleteLatency;
  const FabricConfig Fabric;
  const uint32_t NumArrays;
  const uint32_t AccRawCycles;
  const uint32_t ZeroCycles;
  const uint32_t MoveCyclesPerUop;
  const uint32_t MoveLatency;
  const uint32_t TileWriteLatency;
  const uint32_t TileReadLatency;
  const uint32_t LsuFifoEntries;
  const bool CtrlWaitsForDrain;
  const double AreaPerPe;
  const double TileStorageArea;
  const double EnergyPerMac;
  const double EnergyPerSubtile;
  const double EnergyPerBeat;

  ResourcePool Cmd;    // Uop issue into the engine.
  ResourcePool Arrays; // MAC arrays.
  ResourcePool Ctrl;   // Controller: vtzero, moves, tile writes from loads.
  VectorLsuInterface *Lsu = nullptr;
  VectorRegisterInterface *VectorRegs = nullptr;

  std::array<Cycle, 16> TileReady{};    // Per tile mt0-mt15.
  std::array<Cycle, 16> TileMacReady{}; // A MAC into the tile may start.
  std::array<Cycle, RegId::NumPerFile> OwnVReady{}; // vtmv.v.t results.
  Cycle LastCmdStart = 0;                           // In-order uop issue.
  // Reservation-station uops still in it or popped at or after the last
  // push: {push cycle, pop cycle}; the pop is NoPop until scheduled.
  std::deque<std::pair<Cycle, Cycle>> RsUops;
  static constexpr Cycle NoPop = ~Cycle{0};
  Cycle HeadFree = 0;   // The controller's head uop is released.
  Cycle LastDone = 0;   // Latest result of any instruction.
  Cycle ArrayDrain = 0; // Latest MAC result.
  // Pop cycles of the last LsuFifoEntries beats through each FIFO.
  std::deque<Cycle> LoadFifoPops;  // lsu2vme: the controller pops.
  std::deque<Cycle> StoreFifoPops; // vme2lsu: the LSU pops.
  MatrixStats Stats;

  sparta::DataInPort<InstRef> InInstsPort{&unit_port_set_, "in_insts", 1};
  sparta::DataOutPort<uint32_t> OutCreditsPort{&unit_port_set_, "out_credits"};
  // Instructions arrive in PortUpdate; they are timed in Tick, after the
  // vector module has booked every instruction dispatched in the same cycle.
  sparta::PayloadEvent<InstRef, sparta::SchedulingPhase::Tick> ExecuteEvent{
      &unit_event_set_, "ev_execute",
      CREATE_SPARTA_HANDLER_WITH_DATA(Matrix, executeInst, InstRef)};
  sparta::Counter NumExecuted{&unit_stat_set_, "num_executed",
                              "Matrix instructions executed",
                              sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf

#endif // REEF_PERF_MATRIX_MATRIX_H
