//===- Matrix.cpp - The matrix module -------------------------------------===//

#include "matrix/Matrix.h"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

using namespace reef_perf;

/// Bytes in one tile row of 8-bit elements, and in one memory beat (the
/// 128-bit DTCM bus).
static constexpr uint32_t SubtileBytes = 16;

static FabricConfig getFabric(const Matrix::MatrixParameterSet *Params) {
  FabricConfig F;
  F.Rows = Params->mtx_pe_rows;
  F.Cols = Params->mtx_pe_cols;
  F.MacsPerPeInt8 = Params->mtx_macs_per_pe_int8;
  F.MacsPerPeBf16 = Params->mtx_macs_per_pe_bf16;
  F.MacsPerPeFp32 = Params->mtx_macs_per_pe_fp32;
  F.Flow = parseDataflow(Params->mtx_dataflow);
  F.RowOperand = parseDelivery(Params->mtx_row_operand);
  F.ColOperand = parseDelivery(Params->mtx_col_operand);
  F.HopLatency = Params->mtx_hop_latency;
  F.Reduce = parseReduction(Params->mtx_reduction);
  F.TreeLevelLatency = Params->mtx_tree_level_latency;
  F.PipelineDepth = Params->mtx_pe_pipeline_depth;
  F.AccumulateLatency = Params->mtx_accumulate_latency;
  F.StationaryLoadCycles = Params->mtx_stationary_load_cycles;
  F.DoubleBufferStationary = Params->mtx_double_buffer_stationary;
  F.PipelineAcrossOps = Params->mtx_pipeline_across_ops;
  F.ClockDivider = Params->mtx_fabric_clock_divider;
  // Fails early on a fabric that cannot run int8.
  getMacCost(F, 1, 1, 1, MatrixType::Int8);
  return F;
}

Matrix::Matrix(sparta::TreeNode *Node, const MatrixParameterSet *Params)
    : sparta::Unit(Node, name), QueueEntries(Params->mtx_queue_entries),
      CmdLatency(Params->mtx_cmd_latency),
      UopsPerCycle(Params->mtx_uops_per_cycle),
      CompleteLatency(Params->mtx_complete_latency), Fabric(getFabric(Params)),
      NumArrays(Params->mtx_num_arrays),
      AccRawCycles(Params->mtx_acc_raw_cycles),
      ZeroCycles(Params->mtx_zero_cycles),
      MoveCyclesPerUop(Params->mtx_move_cycles_per_uop),
      MoveLatency(Params->mtx_move_latency),
      TileWriteLatency(Params->mtx_tile_write_latency),
      TileReadLatency(Params->mtx_tile_read_latency),
      CtrlWaitsForDrain(Params->mtx_ctrl_waits_for_array_drain),
      AreaPerPe(Params->mtx_area_per_pe_um2),
      TileStorageArea(Params->mtx_tile_storage_area_um2),
      EnergyPerMac(Params->mtx_energy_per_mac_pj),
      EnergyPerSubtile(Params->mtx_energy_per_subtile_access_pj),
      EnergyPerBeat(Params->mtx_energy_per_mem_beat_pj), Cmd("matrix_cmd", 1),
      Arrays("matrix_array", Params->mtx_num_arrays), Ctrl("matrix_ctrl", 1) {
  if (QueueEntries == 0 || UopsPerCycle == 0)
    throw std::invalid_argument(
        "mtx_queue_entries and mtx_uops_per_cycle must be at least 1");
  InInstsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Matrix, receiveInst, InstRef));
  sparta::StartupEvent(Node, CREATE_SPARTA_HANDLER(Matrix, sendInitialCredits));
}

void Matrix::sendInitialCredits() { OutCreditsPort.send(QueueEntries); }

void Matrix::receiveInst(const InstRef &I) {
  ExecuteEvent.preparePayload(I)->schedule(); // Tick phase, this cycle.
}

uint32_t Matrix::getUops(const Inst &I) {
  switch (I.Mtx.Op) {
  case MatrixOp::Mac:
    // Always 4: uop_index_max = 3 (rvv_backend_decode_unit_ari.sv:564-575,
    // :2059-2073).
    return 4;
  case MatrixOp::TileLoad:
  case MatrixOp::TileStore: // One per 16-byte beat (_lsu.sv:3048-3070).
    return std::max<uint32_t>(
        1, (uint32_t{I.Mtx.TN} * I.Mtx.ElemBytes + SubtileBytes - 1) /
               SubtileBytes);
  case MatrixOp::MoveToTile:
  case MatrixOp::MoveToVector: // 1/2/4 for SEW 8/16/32: one per register.
    return std::max<uint32_t>(1, I.Mtx.ElemBytes);
  case MatrixOp::Zero:
    return 1;
  case MatrixOp::Discard:
  case MatrixOp::Nop:
  case MatrixOp::None:
  case MatrixOp::NumOps:
    break;
  }
  return 0;
}

Cycle Matrix::getVectorSourcesReady(const Inst &I) const {
  Cycle Ready = 0;
  for (RegId Reg : I.Srcs) {
    if (Reg.getFile() != RegFile::V)
      continue;
    unsigned Idx = Reg.getIndex();
    Ready = std::max(Ready, OwnVReady.at(Idx));
    if (VectorRegs)
      Ready = std::max(Ready, VectorRegs->getRegisterReady(Idx));
  }
  return Ready;
}

Cycle Matrix::getTilesReady(const Inst &I, const std::array<Cycle, 16> &Ready) {
  Cycle Latest = 0;
  for (uint32_t T = I.Mtx.FirstTile;
       T < uint32_t{I.Mtx.FirstTile} + I.Mtx.NumTiles && T < Ready.size(); ++T)
    Latest = std::max(Latest, Ready.at(T));
  return Latest;
}

void Matrix::setTilesReady(const Inst &I, std::array<Cycle, 16> &Tiles,
                           Cycle Ready) {
  for (uint32_t T = I.Mtx.FirstTile;
       T < uint32_t{I.Mtx.FirstTile} + I.Mtx.NumTiles && T < Tiles.size(); ++T)
    Tiles.at(T) = Ready;
}

Cycle Matrix::bookBeats(const Inst &I, Cycle Earliest) {
  sparta_assert(Lsu, "the core's LSU was not connected to the matrix module");
  std::vector<std::span<const MemAccess>> Beats =
      splitByRegister(I.Mem, SubtileBytes);
  if (Beats.empty())
    Beats.emplace_back(I.Mem);
  Cycle Done = Earliest;
  for (size_t B = 0; B < Beats.size(); ++B) {
    VectorLsuRequest Request;
    Request.IsStore = I.Mtx.Op == MatrixOp::TileStore;
    Request.Accesses = Beats[B];
    Request.Earliest = Earliest;
    // Dispatch gave the instruction an LSU queue entry; the LSU frees it
    // when the first beat starts.
    Request.FirstRegister = B == 0;
    Request.LastRegister = B + 1 == Beats.size();
    Done = std::max(Done, Lsu->accessVectorRegister(Request).Done);
  }
  Stats.MemBeats += Beats.size();
  return Done;
}

void Matrix::executeInst(const InstRef &I) {
  Cycle Now = getClock()->currentCycle();
  ++NumExecuted;
  sparta_assert(I->Mtx.Op != MatrixOp::None && I->Mtx.Op != MatrixOp::NumOps,
                "matrix module got a non-matrix instruction");
  ++Stats.Ops.at(static_cast<size_t>(I->Mtx.Op));

  // 1. Command path: in program order, once vector sources are ready.
  uint32_t Uops = getUops(*I);
  uint64_t IssueCycles =
      std::max<uint64_t>(1, (uint64_t{Uops} + UopsPerCycle - 1) / UopsPerCycle);
  Cycle CmdEarliest = std::max(
      {Cycle{Now + CmdLatency}, getVectorSourcesReady(*I), LastCmdStart});
  Cycle CmdStart = Cmd.reserve(CmdEarliest, IssueCycles);
  Cycle CmdDone = CmdStart + IssueCycles;
  LastCmdStart = CmdStart;
  OutCreditsPort.send(1, CmdStart - Now); // Leaves the queue as it issues.

  // 2. Execution. CoralNPU's controller serves one head uop at a time
  // (zvt_ctrl.sv), so an instruction starts only once the one before has
  // released the head: a MAC when its compute steps end (canStart && cntCr,
  // zvt_pe_array.sv:209), vtzero after its counter (zvt_ctrl.sv:613), loads,
  // stores and moves after one controller cycle per uop.
  const MatrixInfo &M = I->Mtx;
  // A MAC reads its accumulator from the pipeline, so after another MAC
  // into the same tile it waits only for TileMacReady: hitRaw stalls the
  // first step while the earlier MAC's first step is in the multiply stages
  // (zvt_pe_array.sv:195-207).
  Cycle TilesReady =
      getTilesReady(*I, M.Op == MatrixOp::Mac ? TileMacReady : TileReady);
  Cycle Earliest = std::max({CmdDone, TilesReady, HeadFree});
  // Every controller op except a MAC waits for the array to drain (!peBusy:
  // zvt_ctrl.sv:105-106, :130-134).
  bool UsesCtrl = M.Op == MatrixOp::Zero || M.Op == MatrixOp::MoveToTile ||
                  M.Op == MatrixOp::MoveToVector ||
                  M.Op == MatrixOp::TileLoad || M.Op == MatrixOp::TileStore;
  if (UsesCtrl && CtrlWaitsForDrain)
    Earliest = std::max(Earliest, ArrayDrain);
  Cycle Start = Earliest;
  Cycle Done = Earliest;
  uint64_t SliceSubtiles =
      (uint64_t{M.TN} * M.ElemBytes + SubtileBytes - 1) / SubtileBytes;
  switch (M.Op) {
  case MatrixOp::Mac: {
    FabricCost Cost = getMacCost(Fabric, M.TM, M.TN, M.TK, M.Type);
    Start = Arrays.reserve(Earliest, Cost.Occupancy);
    Done = Start + Cost.Latency;
    HeadFree = Start + Cost.Busy;
    setTilesReady(*I, TileReady, std::max(Done, getTilesReady(*I, TileReady)));
    setTilesReady(*I, TileMacReady,
                  AccRawCycles == 0 ? Done : Start + AccRawCycles);
    ArrayDrain = std::max(ArrayDrain, Done);
    Stats.UsefulMacs += Cost.UsefulMacs;
    Stats.CapacityMacs += Cost.CapacityMacs;
    Stats.ArrayBusyCycles += Cost.Busy;
    // The accumulator rows touched are read and written once.
    uint64_t AccSubtiles =
        (uint64_t{M.TM} * M.TN * 4 + SubtileBytes - 1) / SubtileBytes;
    Stats.SubtileReads += AccSubtiles;
    Stats.SubtileWrites += AccSubtiles;
    break;
  }
  case MatrixOp::Zero:
    Start = Ctrl.reserve(Earliest, ZeroCycles);
    Done = Start + ZeroCycles;
    HeadFree = Done;
    setTilesReady(*I, TileReady, Done);
    setTilesReady(*I, TileMacReady, Done);
    Stats.SubtileWrites += uint64_t{M.NumTiles} * 16; // 16 x 16 bytes each.
    break;
  case MatrixOp::MoveToTile:
  case MatrixOp::MoveToVector: {
    uint64_t Busy = uint64_t{Uops} * MoveCyclesPerUop;
    Start = Ctrl.reserve(Earliest, Busy);
    HeadFree = Start + Busy;
    Done = HeadFree + MoveLatency;
    if (M.Op == MatrixOp::MoveToTile) {
      setTilesReady(*I, TileReady, Done);
      setTilesReady(*I, TileMacReady, Done);
      Stats.SubtileWrites += SliceSubtiles;
    } else {
      for (RegId Reg : I->Dsts)
        if (Reg.getFile() == RegFile::V)
          OwnVReady.at(Reg.getIndex()) = Done;
      Stats.SubtileReads += SliceSubtiles;
    }
    break;
  }
  case MatrixOp::TileLoad: {
    // The LSU half was dispatched with the instruction (CoralNPU sends tile
    // loads to the LSU and the RVV backend at once), so it may start now.
    // The controller writes one beat per cycle, the last once it arrives.
    Cycle LastData = bookBeats(*I, Now);
    Cycle FirstWrite = LastData > Uops - 1 ? LastData - (Uops - 1) : 0;
    Start = Ctrl.reserve(std::max(Earliest, FirstWrite), Uops);
    HeadFree = Start + Uops;
    Done = Start + Uops - 1 + TileWriteLatency;
    setTilesReady(*I, TileReady, Done);
    setTilesReady(*I, TileMacReady, Done);
    Stats.SubtileWrites += SliceSubtiles;
    break;
  }
  case MatrixOp::TileStore:
    // The controller reads one beat per cycle into the vme2lsu path.
    Start = Ctrl.reserve(Earliest, Uops);
    HeadFree = Start + Uops;
    Done = bookBeats(*I, Start + TileReadLatency);
    Stats.SubtileReads += SliceSubtiles;
    break;
  case MatrixOp::Discard:
    // Waits for every matrix result; CoralNPU also waits for an empty
    // retirement buffer and an idle LSU, which is not modeled.
    Start = std::max(Earliest, LastDone);
    Done = Start + 1;
    HeadFree = Done;
    break;
  case MatrixOp::Nop:
    // Retired by the RVV decoder; it never reaches the engine.
    Start = CmdDone;
    Done = CmdDone;
    break;
  case MatrixOp::None:
  case MatrixOp::NumOps:
    break;
  }
  LastDone = std::max(LastDone, Done);

  I->IssueCycle = CmdStart;
  I->ResultReadyCycle = Done;
  I->CompleteCycle = Done + CompleteLatency;
}

double Matrix::getAreaUm2() const {
  if (AreaPerPe <= 0.0 && TileStorageArea <= 0.0)
    return -1.0;
  return AreaPerPe * NumArrays * Fabric.Rows * Fabric.Cols + TileStorageArea;
}

double Matrix::getEnergyPj() const {
  if (EnergyPerMac <= 0.0 && EnergyPerSubtile <= 0.0 && EnergyPerBeat <= 0.0)
    return -1.0;
  return EnergyPerMac * static_cast<double>(Stats.UsefulMacs) +
         EnergyPerSubtile *
             static_cast<double>(Stats.SubtileReads + Stats.SubtileWrites) +
         EnergyPerBeat * static_cast<double>(Stats.MemBeats);
}
