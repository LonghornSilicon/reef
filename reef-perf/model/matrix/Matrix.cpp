//===- Matrix.cpp - The matrix module -------------------------------------===//

#include "matrix/Matrix.h"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

using namespace reef_perf;

/// Bytes in one tile row of 8-bit elements, and in one memory beat (the
/// 128-bit DTCM bus).
static constexpr uint32_t SubtileBytes = 16;

/// \returns \p Value. \throws std::invalid_argument naming \p Name if it is 0.
static uint32_t atLeastOne(uint32_t Value, const char *Name) {
  if (Value == 0)
    throw std::invalid_argument(std::string(Name) + " must be at least 1");
  return Value;
}

static FabricConfig getFabric(const Matrix::MatrixParameterSet *Params) {
  FabricConfig F;
  F.Rows = Params->mtx_pe_rows;
  F.Cols = Params->mtx_pe_cols;
  F.MacsPerPeInt8 = Params->mtx_macs_per_pe_int8;
  F.MacsPerPeBf16 = Params->mtx_macs_per_pe_bf16;
  F.MacsPerPeFp32 = Params->mtx_macs_per_pe_fp32;
  F.Flow = parseDataflow(Params->mtx_dataflow);
  F.RowOperand = parseDelivery(Params->mtx_row_operand, "mtx_row_operand");
  F.ColOperand = parseDelivery(Params->mtx_col_operand, "mtx_col_operand");
  F.HopLatency = Params->mtx_hop_latency;
  F.Reduce = parseReduction(Params->mtx_reduction);
  F.TreeLevelLatency = Params->mtx_tree_level_latency;
  F.PipelineDepth = Params->mtx_pe_pipeline_depth;
  F.AccumulateLatency = Params->mtx_accumulate_latency;
  F.StationaryLoadCycles = Params->mtx_stationary_load_cycles;
  F.DoubleBufferStationary = Params->mtx_double_buffer_stationary;
  F.PipelineAcrossOps = Params->mtx_pipeline_across_ops;
  F.ClockDivider = Params->mtx_fabric_clock_divider;
  F.Map = parseMapping(Params->mtx_mapping);
  checkFabricConfig(F);
  return F;
}

Matrix::Matrix(sparta::TreeNode *Node, const MatrixParameterSet *Params)
    : sparta::Unit(Node, name),
      QueueEntries(atLeastOne(Params->mtx_queue_entries, "mtx_queue_entries")),
      CmdLatency(Params->mtx_cmd_latency),
      UopsPerCycle(
          atLeastOne(Params->mtx_uops_per_cycle, "mtx_uops_per_cycle")),
      UopBufferEntries(Params->mtx_uop_buffer_entries),
      CompleteLatency(Params->mtx_complete_latency), Fabric(getFabric(Params)),
      NumArrays(atLeastOne(Params->mtx_num_arrays, "mtx_num_arrays")),
      AccRawCycles(Params->mtx_acc_raw_cycles),
      ZeroCycles(Params->mtx_zero_cycles),
      MoveCyclesPerUop(Params->mtx_move_cycles_per_uop),
      MoveLatency(Params->mtx_move_latency),
      TileWriteLatency(Params->mtx_tile_write_latency),
      TileReadLatency(Params->mtx_tile_read_latency),
      LsuFifoEntries(
          atLeastOne(Params->mtx_lsu_fifo_entries, "mtx_lsu_fifo_entries")),
      CtrlWaitsForDrain(Params->mtx_ctrl_waits_for_array_drain),
      AreaPerPe(Params->mtx_area_per_pe_um2),
      TileStorageArea(Params->mtx_tile_storage_area_um2),
      EnergyPerMac(Params->mtx_energy_per_mac_pj),
      EnergyPerSubtile(Params->mtx_energy_per_subtile_access_pj),
      EnergyPerBeat(Params->mtx_energy_per_mem_beat_pj), Cmd("matrix_cmd", 1),
      Arrays("matrix_array", NumArrays), Ctrl("matrix_ctrl", 1) {
  if (UopBufferEntries < 4)
    throw std::invalid_argument("mtx_uop_buffer_entries must hold a MAC's 4 "
                                "uops");
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

/// \returns \p I's 16-byte beats, in order.
static std::vector<std::span<const MemAccess>> getBeats(const Inst &I) {
  std::vector<std::span<const MemAccess>> Beats =
      splitByRegister(I.Mem, SubtileBytes);
  if (Beats.empty())
    Beats.emplace_back(I.Mem);
  return Beats;
}

Cycle Matrix::getFifoFree(const std::deque<Cycle> &Pops) const {
  // A push may share a cycle with the pop that frees its entry
  // (multi_fifo.sv:112, entry_count_now).
  return Pops.size() < LsuFifoEntries ? 0 : Pops.front();
}

void Matrix::recordPop(std::deque<Cycle> &Pops, Cycle When) const {
  Pops.push_back(When);
  if (Pops.size() > LsuFifoEntries)
    Pops.pop_front();
}

Matrix::Transfer Matrix::runTileLoad(const Inst &I, Cycle Now,
                                     Cycle CtrlEarliest) {
  sparta_assert(Lsu, "the core's LSU was not connected to the matrix module");
  std::vector<std::span<const MemAccess>> Beats = getBeats(I);
  Transfer T;
  for (size_t B = 0; B < Beats.size(); ++B) {
    VectorLsuRequest Request;
    Request.IsStore = false;
    Request.Accesses = Beats[B];
    Request.IsTile = true; // The LSU may cost it apart (lsu_tile_cost).
    // The LSU takes a beat once lsu2vme has room for it. The RTL holds the
    // finished beat in the LSU instead; starting it later is the same stall
    // without modeling the LSU's hold.
    Request.Earliest = std::max(Now, getFifoFree(LoadFifoPops));
    // Dispatch gave the instruction an LSU queue entry; the LSU frees it
    // when the first beat starts.
    Request.FirstRegister = B == 0;
    Request.LastRegister = B + 1 == Beats.size();
    Cycle Data = Lsu->accessVectorRegister(Request).Done;
    // The controller pops one beat per cycle and writes it to the tile
    // (isLsu2Vme, zvt_ctrl.sv:106), no earlier than the cycle after the LSU
    // pushed it: the FIFO's count is a register (multi_fifo.sv:107), and on
    // the VME RTL a beat pushed in cycle t is popped in t + 1.
    Cycle Pop = Ctrl.reserve(
        std::max({CtrlEarliest, Data + 1, B == 0 ? Cycle{0} : T.CtrlLast + 1}),
        1);
    if (B == 0)
      T.CtrlFirst = Pop;
    T.CtrlLast = Pop;
    recordPop(LoadFifoPops, Pop);
  }
  Stats.MemBeats += Beats.size();
  return T;
}

Matrix::Transfer Matrix::runTileStore(const Inst &I, Cycle CtrlEarliest) {
  sparta_assert(Lsu, "the core's LSU was not connected to the matrix module");
  std::vector<std::span<const MemAccess>> Beats = getBeats(I);
  Transfer T;
  for (size_t B = 0; B < Beats.size(); ++B) {
    // The controller reads one beat per cycle and pushes it into vme2lsu
    // TileReadLatency later, once the FIFO has room (isVme2Lsu,
    // zvt_ctrl.sv:105).
    Cycle Free = getFifoFree(StoreFifoPops);
    Cycle Room = Free > TileReadLatency ? Free - TileReadLatency : 0;
    Cycle Read = Ctrl.reserve(
        std::max({CtrlEarliest, Room, B == 0 ? Cycle{0} : T.CtrlLast + 1}), 1);
    if (B == 0)
      T.CtrlFirst = Read;
    T.CtrlLast = Read;
    VectorLsuRequest Request;
    Request.IsStore = true;
    Request.Accesses = Beats[B];
    Request.IsTile = true; // The LSU may cost it apart (lsu_tile_cost).
    Request.Earliest = Read + TileReadLatency;
    Request.FirstRegister = B == 0;
    Request.LastRegister = B + 1 == Beats.size();
    VectorLsuResponse Response = Lsu->accessVectorRegister(Request);
    recordPop(StoreFifoPops, Response.Start);
    T.LsuDone = std::max(T.LsuDone, Response.Done);
  }
  Stats.MemBeats += Beats.size();
  return T;
}

std::pair<Cycle, Cycle> Matrix::pushUops(uint32_t Uops, Cycle Earliest) {
  Cycle T = Earliest;
  if (!RsUops.empty())
    T = std::max(T, RsUops.back().first);
  if (Uops == 0)
    return {T, T};
  // Uops popped before the last push no longer change any count below.
  while (!RsUops.empty() && RsUops.front().second < RsUops.back().first)
    RsUops.pop_front();
  Cycle First = 0;
  for (uint32_t U = 0; U < Uops; ++U) {
    for (;; ++T) {
      // Each push lane checks the room left at the start of the cycle,
      // before that cycle's pops (almost_full per lane, multi_fifo.sv;
      // rs_ready_zvt2dp, rvv_backend.sv:797-807).
      uint64_t In = 0;
      uint64_t PushedNow = 0;
      for (const auto &[Push, Pop] : RsUops) {
        if (Push < T && Pop >= T)
          ++In;
        if (Push == T)
          ++PushedNow;
      }
      if (PushedNow < UopsPerCycle && In + PushedNow < UopBufferEntries)
        break;
    }
    if (U == 0)
      First = T;
    RsUops.emplace_back(T, NoPop);
    if (Cmd.getNextFree(T) == T)
      Cmd.reserve(T, 1); // Pool statistics: cycles with a push.
  }
  return {First, T};
}

void Matrix::popUops(uint32_t Uops, Cycle Last, bool Together) {
  sparta_assert(RsUops.size() >= Uops, "popping uops that were not pushed");
  for (uint32_t U = 0; U < Uops; ++U) {
    Cycle Pop = Together ? Last : Last - (Uops - 1 - U);
    RsUops[RsUops.size() - Uops + U].second = Pop;
  }
}

void Matrix::executeInst(const InstRef &I) {
  Cycle Now = getClock()->currentCycle();
  ++NumExecuted;
  sparta_assert(I->Mtx.Op != MatrixOp::None && I->Mtx.Op != MatrixOp::NumOps,
                "matrix module got a non-matrix instruction");
  ++Stats.Ops.at(static_cast<size_t>(I->Mtx.Op));

  // 1. Command path: uops enter the reservation station in program order,
  // once vector sources are ready; a uop can leave the cycle after it
  // enters.
  uint32_t Uops = getUops(*I);
  Cycle CmdEarliest = std::max(
      {Cycle{Now + CmdLatency}, getVectorSourcesReady(*I), LastCmdStart});
  auto [CmdStart, LastPush] = pushUops(Uops, CmdEarliest);
  Cycle CmdDone = LastPush + 1;
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
  bool UsesCtrl = M.Op == MatrixOp::Zero || M.Op == MatrixOp::MoveToTile ||
                  M.Op == MatrixOp::MoveToVector ||
                  M.Op == MatrixOp::TileLoad || M.Op == MatrixOp::TileStore;
  // A MAC needs all its uops at the head (isPe over the ZVT_LMUL uop lanes,
  // zvt_ctrl.sv:108-119). The controller serves the other ops one uop per
  // cycle from the first uop on; later uops issue faster than that, so they
  // never wait (VME waveforms of move_tv_rob and move_vt_use: dispatch to
  // first uop 5 cycles for a 4-uop move, 6 for a MAC).
  Cycle CmdReady = UsesCtrl ? CmdStart + 1 : CmdDone;
  Cycle Earliest = std::max({CmdReady, TilesReady, HeadFree});
  // Every controller op except a MAC waits for the array to drain (!peBusy:
  // zvt_ctrl.sv:105-106, :130-134).
  // The controller acts the cycle after the last MAC's tile write: on the
  // VME RTL, peBusy falls 13 cycles after a MAC's first step and the first
  // tile-load pop behind it comes at 14 (waveform of tile_load_behind_mac).
  if (UsesCtrl && CtrlWaitsForDrain)
    Earliest = std::max(Earliest, ArrayDrain + 1);
  Cycle Start = Earliest;
  Cycle Done = Earliest;
  uint64_t SliceSubtiles =
      (uint64_t{M.TN} * M.ElemBytes + SubtileBytes - 1) / SubtileBytes;
  switch (M.Op) {
  case MatrixOp::Mac: {
    FabricCost Cost = getMacCost(Fabric, M.TM, M.TN, M.TK, M.Type);
    Start = Arrays.reserve(Earliest, Cost.Occupancy);
    Done = Start + Cost.Latency;
    // CoralNPU's one array holds the controller's head until its compute
    // steps end (canStart && cntCr). With more arrays the controller hands
    // the MAC to a free one and moves on the next cycle, so MACs on
    // different arrays overlap; the Arrays pool limits them (an assumption
    // for the sweep, no RTL).
    HeadFree = NumArrays == 1 ? Start + Cost.Busy : Start + 1;
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
        if (Reg.getFile() == RegFile::V) {
          OwnVReady.at(Reg.getIndex()) = Done;
          // Vector readers wait for it in the vector module.
          if (VectorRegs)
            VectorRegs->setRegisterReady(Reg.getIndex(), Done);
        }
      Stats.SubtileReads += SliceSubtiles;
    }
    break;
  }
  case MatrixOp::TileLoad: {
    // The LSU half was dispatched with the instruction (CoralNPU sends tile
    // loads to the LSU and the RVV backend at once), so it may start now;
    // the 2-deep lsu2vme FIFO (zvt.sv:163-206) keeps it at most that many
    // beats ahead of the controller.
    Transfer T = runTileLoad(*I, Now, Earliest);
    Start = T.CtrlFirst;
    HeadFree = T.CtrlLast + 1;
    Done = T.CtrlLast + TileWriteLatency;
    setTilesReady(*I, TileReady, Done);
    setTilesReady(*I, TileMacReady, Done);
    Stats.SubtileWrites += SliceSubtiles;
    break;
  }
  case MatrixOp::TileStore: {
    // The controller stalls when the LSU is LsuFifoEntries beats behind.
    Transfer T = runTileStore(*I, Earliest);
    Start = T.CtrlFirst;
    HeadFree = T.CtrlLast + 1;
    Done = T.LsuDone;
    Stats.SubtileReads += SliceSubtiles;
    break;
  }
  case MatrixOp::Discard:
    // The engine does nothing with vtdiscard: the RVV decoder rejects its
    // encoding (no check_special case under VWRXUNARY0,
    // rvv_backend_decode_unit_ari.sv:3600-3607) and retires it, so it never
    // holds the controller. Its waiting is the core's dispatch interlock
    // (empty retirement buffer, RVV idle, LSU idle; Decode.scala:601-607).
    // The engine's share of "RVV idle" is kept: it completes no earlier
    // than every older matrix result. The rest is a core rule, not modeled.
    Start = std::max(CmdDone, LastDone);
    Done = Start;
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
  if (Uops != 0)
    // A MAC leaves with its last step (peCmdRdy = canStart && cntCr,
    // zvt_pe_array.sv:206); the controller takes one uop per cycle.
    popUops(Uops, std::max(HeadFree, Start + 1) - 1, M.Op == MatrixOp::Mac);
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
