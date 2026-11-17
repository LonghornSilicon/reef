//===- Lsu.cpp - The load/store unit --------------------------------------===//

#include "core/Lsu.h"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <algorithm>
#include <cstdint>

using namespace reef_perf;

Lsu::Lsu(sparta::TreeNode *Node, const LsuParameterSet *Params)
    : sparta::Unit(Node, name), QueueEntries(Params->lsu_queue_entries),
      ScalarLatencyOverhead(Params->lsu_scalar_latency_overhead),
      ScalarOccupancyOverhead(Params->lsu_scalar_occupancy_overhead),
      VectorLoadRegOverhead(Params->lsu_vector_load_reg_overhead),
      VectorStoreRegOverhead(Params->lsu_vector_store_reg_overhead),
      VectorSwitchCycles(Params->lsu_vector_switch_cycles),
      TileCost(Params->lsu_tile_cost),
      TileInstOverhead(Params->lsu_tile_inst_overhead),
      TileBeatOverhead(Params->lsu_tile_beat_overhead),
      TileSwitchCycles(Params->lsu_tile_switch_cycles), V3(Params->lsu_v3),
      V3SkipHandshake(Params->lsu_v3_skip_handshake),
      V3Tile(Params->lsu_v3_tile), V3SetupCycles(Params->lsu_v3_setup_cycles),
      V3ReturnCycles(Params->lsu_v3_return_cycles),
      V3ScalarStoreCycles(Params->lsu_v3_scalar_store_cycles),
      V3HandshakeCycles(Params->lsu_v3_handshake_cycles),
      V3BackendLead(Params->lsu_v3_backend_lead), Pipe("lsu", 1) {
  InInstsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Lsu, receiveInst, InstRef));
  sparta::StartupEvent(Node, CREATE_SPARTA_HANDLER(Lsu, sendInitialCredits));
}

void Lsu::sendInitialCredits() { OutCreditsPort.send(QueueEntries); }

void Lsu::receiveInst(const InstRef &I) {
  sparta_assert(Mem, "memory was not connected to the LSU");
  sparta_assert(!I->isVector(), "vector memory instructions come through the "
                                "vector module");
  Cycle Now = getClock()->currentCycle();
  ++NumExecuted;
  if (V3) {
    receiveInstV3(I);
    return;
  }

  MemRequest Request;
  Request.Requester = RequesterKind::LSU;
  Request.Accesses = I->Mem;
  Request.Earliest = Pipe.getNextFree(Now);
  MemResponse Response = Mem->access(Request);
  Cycle Start = Response.Start;
  // Memory reports the physical transfer. Scalar instructions also traverse
  // LSU pipeline stages; vector registers add their path delay separately in
  // accessVectorRegister().
  Cycle Reserved =
      Pipe.reserve(Start, Response.Occupancy + ScalarOccupancyOverhead);
  sparta_assert(Reserved == Start, "LSU busy at the memory's start");
  I->IssueCycle = Start;
  I->ResultReadyCycle = Start + Response.Latency + ScalarLatencyOverhead - 1;
  I->CompleteCycle = Start + Response.Latency + ScalarLatencyOverhead;

  // The instruction leaves the queue when it starts.
  OutCreditsPort.send(1, Start - Now);
}

VectorLsuResponse Lsu::accessVectorRegister(const VectorLsuRequest &Request) {
  sparta_assert(Mem, "memory was not connected to the LSU");
  ++NumVectorRegisters;
  // Tile beats take the LsuV3 path only with lsu_v3_tile; otherwise they
  // keep M3's path (lsu_tile_cost, or a vector register's cost).
  if (V3 && (!Request.IsTile || V3Tile))
    return accessVectorRegisterV3(Request);
  uint64_t Overhead =
      Request.IsStore ? VectorStoreRegOverhead : VectorLoadRegOverhead;
  uint64_t SwitchCycles = VectorSwitchCycles;
  // A Zvt tile beat takes its own cost when lsu_tile_cost is on: an overhead
  // once per instruction, not per beat (docs/lsu-tile-beats.md). Only the
  // matrix module sets IsTile, so vector and scalar timing never take this
  // path.
  if (Request.IsTile && TileCost) {
    Overhead =
        TileBeatOverhead + (Request.FirstRegister ? TileInstOverhead : 0);
    SwitchCycles = TileSwitchCycles;
  }
  MemRequest Access;
  Access.Requester = RequesterKind::LSU;
  Access.Accesses = Request.Accesses;
  Access.Earliest = Pipe.getNextFree(Request.Earliest) + Overhead;
  MemResponse Response = Mem->access(Access);

  VectorLsuResponse Result;
  Result.Start = Response.Start - Overhead;
  Cycle BusyUntil = Response.Start + Response.Occupancy;
  if (Request.LastRegister)
    BusyUntil += SwitchCycles;
  Cycle Reserved = Pipe.reserve(Result.Start, BusyUntil - Result.Start);
  sparta_assert(Reserved == Result.Start, "LSU busy at the memory's start");
  // Dispatch gave the instruction an LSU queue entry; it leaves the queue when
  // its first register starts.
  if (Request.FirstRegister) {
    Cycle Now = getClock()->currentCycle();
    sparta_assert(Result.Start >= Now);
    OutCreditsPort.send(1, Result.Start - Now);
  }
  // The LSU hands the data over the cycle the last line is read, one cycle
  // before the 1-cycle SRAM's data is usable (RTL waveform of vec_add_m1: a
  // load is taken in A, its data is handed back in A + 3 and usable in A + 5).
  Result.Done = Response.Start + Response.Latency - 1;
  // A sender that cannot take the register yet (Request.ReceiverReady) leaves
  // it, and the slot, in the LSU until it can. 0, as the vector module and
  // the matrix module by default send it, never holds.
  if (Request.ReceiverReady > Result.Done) {
    Cycle Free = Pipe.getNextFree(0);
    if (Request.ReceiverReady + 1 > Free)
      Pipe.reserve(Free, Request.ReceiverReady + 1 - Free);
    Result.Done = Request.ReceiverReady;
  }
  return Result;
}

// LsuV3, from waveforms of the 8a5e1b31 RTL (docs/pinned.md, "LsuV3"). The
// slot runs one instruction at a time, in dispatch order. A load reads the
// bus, hands its data back lsu_v3_return_cycles later, and the next
// instruction's bus cycle follows that.

void Lsu::receiveInstV3(const InstRef &I) {
  Cycle Now = getClock()->currentCycle();
  MemRequest Request;
  Request.Requester = RequesterKind::LSU;
  Request.Accesses = I->Mem;
  Request.Earliest = Pipe.getNextFree(Now + V3SetupCycles);
  MemResponse Response = Mem->access(Request);
  Cycle Bus = Response.Start;
  bool IsStore =
      I->Class == InstClass::STORE || I->Class == InstClass::FP_STORE;
  Cycle Done = Bus + Response.Latency - 1 +
               (IsStore ? V3ScalarStoreCycles : V3ReturnCycles);
  Cycle Reserved = Pipe.reserve(Bus, Done + 1 - Bus);
  sparta_assert(Reserved == Bus, "LSU busy at the memory's start");
  I->IssueCycle = Bus;
  // The retirement buffer retires it in the cycle the value is written
  // (waveform of lsu/lw_rob: dispatched in D, retired in D + 4).
  I->ResultReadyCycle = Done;
  I->CompleteCycle = Done;
  OutCreditsPort.send(1, Bus - Now);
}

VectorLsuResponse Lsu::accessVectorRegisterV3(const VectorLsuRequest &Request) {
  bool Continues = !Request.FirstRegister;
  Cycle SlotFree = Continues ? V3NextBus : Pipe.getNextFree(0);
  // A load that needs nothing from the vector backend follows the core's
  // dispatch, not the vector pipeline. Anything else is taken when the
  // backend hands its data over and the slot is free, and reads or writes
  // the bus lsu_v3_handshake_cycles later.
  bool Waits = Request.NeedsVectorData || Request.IsStore || !V3SkipHandshake;
  // A tile load needs nothing from a backend; a tile store waits for the
  // engine's data (Request.Earliest), as a vector store waits for RVV's.
  if (Request.IsTile)
    Waits = Request.IsStore;
  Cycle Earliest = 0;
  if (Waits)
    Earliest = Continues
                   ? std::max(SlotFree, Request.Earliest + V3HandshakeCycles)
                   : std::max(Request.Earliest, SlotFree) + V3HandshakeCycles;
  else
    Earliest = std::max(SlotFree, Request.Dispatched > 0
                                      ? Request.Dispatched + 1 + V3SetupCycles
                                      : Request.Earliest + V3SetupCycles);

  MemRequest Access;
  Access.Requester = RequesterKind::LSU;
  Access.Accesses = Request.Accesses;
  Access.Earliest = Earliest;
  MemResponse Response = Mem->access(Access);
  Cycle Bus = Response.Start;
  V3NextBus = Bus + Response.Occupancy;

  // The backend takes a loaded register lsu_v3_backend_lead cycles before
  // the uop could issue (Request.Earliest).
  VectorLsuResponse Result;
  Result.Start =
      Waits ? Bus - V3HandshakeCycles : std::max(Request.Earliest, Bus);
  // A tile store has no writeback to a backend: it is done when a scalar
  // store would be.
  Result.Done = Bus + Response.Latency - 1 +
                (Request.IsTile && Request.IsStore ? V3ScalarStoreCycles
                                                   : V3ReturnCycles);
  if (Request.Earliest >= V3BackendLead)
    Result.Done = std::max(Result.Done, Request.Earliest - V3BackendLead);
  Result.Done = std::max(Result.Done, Request.ReceiverReady);
  // The slot is the instruction's from its first register until its last
  // one is done; a later register extends the booking.
  Cycle From = Pipe.getNextFree(0);
  if (!Continues)
    From = std::max(From, Waits ? Bus - V3HandshakeCycles : Bus);
  if (Result.Done + 1 > From)
    Pipe.reserve(From, Result.Done + 1 - From);
  if (Request.FirstRegister) {
    Cycle Now = getClock()->currentCycle();
    OutCreditsPort.send(1, std::max(Bus, Now) - Now);
  }
  return Result;
}
