//===- Lsu.cpp - The load/store unit --------------------------------------===//

#include "core/Lsu.h"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

using namespace reef_perf;

/// \returns the LSU's slot count, after checking the LsuV3 design knobs.
/// \throws std::invalid_argument naming a knob with an impossible value.
static uint32_t getSlotCount(const Lsu::LsuParameterSet *Params) {
  if (Params->lsu_v3_slots == 0)
    throw std::invalid_argument("lsu_v3_slots must be at least 1");
  if (Params->lsu_v3_bus_bytes == 0 ||
      Params->lsu_v3_bus_bytes % VectorRegisterBytes != 0)
    throw std::invalid_argument(
        "lsu_v3_bus_bytes must be a nonzero multiple of 16");
  return Params->lsu_v3 ? Params->lsu_v3_slots : 1;
}

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
      V3BackendLead(Params->lsu_v3_backend_lead),
      V3BusBytes(Params->lsu_v3_bus_bytes), Pipe("lsu", getSlotCount(Params)) {
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
// instruction's bus cycle follows that. With lsu_v3_slots above 1, the next
// instruction takes another slot and may use the bus as soon as the previous
// one's last bus cycle is over.

void Lsu::receiveInstV3(const InstRef &I) {
  Cycle Now = getClock()->currentCycle();
  uint32_t Slot = Pipe.getEarliestFree();
  MemRequest Request;
  Request.Requester = RequesterKind::LSU;
  Request.Accesses = I->Mem;
  Request.Earliest = std::max(Now + V3SetupCycles, Pipe.getFreeAt(Slot));
  if (Pipe.getCount() > 1)
    Request.Earliest = std::max(Request.Earliest, V3BusFree);
  MemResponse Response = Mem->access(Request);
  Cycle Bus = Response.Start;
  V3BusFree = Bus + Response.Occupancy;
  bool IsStore =
      I->Class == InstClass::STORE || I->Class == InstClass::FP_STORE;
  Cycle Done = Bus + Response.Latency - 1 +
               (IsStore ? V3ScalarStoreCycles : V3ReturnCycles);
  Cycle Reserved = Pipe.reserveOn(Slot, Bus, Done + 1 - Bus);
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
  if (!Continues) {
    V3Slot = Pipe.getEarliestFree();
    V3BeatOpen = false;
  }
  Cycle SlotFree = Continues ? V3NextBus : Pipe.getFreeAt(V3Slot);
  if (!Continues && Pipe.getCount() > 1)
    SlotFree = std::max(SlotFree, V3BusFree);
  // A load that needs nothing from the vector backend follows the core's
  // dispatch, not the vector pipeline. Anything else is taken when the
  // backend hands its data over and the slot is free, and reads or writes
  // the bus lsu_v3_handshake_cycles later.
  bool Waits = Request.NeedsVectorData || Request.IsStore || !V3SkipHandshake;
  // A tile load needs nothing from a backend; a tile store waits for the
  // engine's data (Request.Earliest), as a vector store waits for RVV's.
  if (Request.IsTile)
    Waits = Request.IsStore;
  Cycle Ready =
      Waits ? Request.Earliest + V3HandshakeCycles
            : (Request.Dispatched > 0 ? Request.Dispatched + 1 + V3SetupCycles
                                      : Request.Earliest + V3SetupCycles);
  Cycle Earliest = 0;
  if (Waits)
    Earliest = Continues
                   ? std::max(SlotFree, Ready)
                   : std::max(Request.Earliest, SlotFree) + V3HandshakeCycles;
  else
    Earliest = std::max(SlotFree, Ready);

  // The register's bus row (lsu_v3_bus_bytes wide), if all its bytes are in
  // one. A register in the row of the instruction's current beat, ready by
  // that beat, rides in it: the bus is wider than one register.
  bool OneRow = false;
  Address Row = 0;
  if (!Request.Accesses.empty()) {
    Address Lo = Request.Accesses.front().Addr;
    Address Hi = Lo;
    for (const MemAccess &A : Request.Accesses) {
      Lo = std::min(Lo, A.Addr);
      Hi = std::max(Hi, A.Addr + std::max<Address>(A.Size, 1) - 1);
    }
    Row = Lo / V3BusBytes;
    OneRow = Hi / V3BusBytes == Row;
  }
  bool Shares = Continues && V3BusBytes > VectorRegisterBytes && V3BeatOpen &&
                OneRow && Row == V3BeatRow && Ready <= V3BeatBus;

  Cycle Bus = 0;
  uint64_t Latency = 1;
  if (Shares) {
    Bus = V3BeatBus;
    Latency = V3BeatLatency;
  } else {
    MemRequest Access;
    Access.Requester = RequesterKind::LSU;
    Access.Accesses = Request.Accesses;
    Access.Earliest = Earliest;
    MemResponse Response = Mem->access(Access);
    Bus = Response.Start;
    Latency = Response.Latency;
    V3NextBus = Bus + Response.Occupancy;
    V3BusFree = V3NextBus;
    V3BeatOpen = OneRow;
    V3BeatBus = Bus;
    V3BeatLatency = Latency;
    V3BeatRow = Row;
  }

  // The backend takes a loaded register lsu_v3_backend_lead cycles before
  // the uop could issue (Request.Earliest).
  VectorLsuResponse Result;
  Result.Start =
      Waits ? Bus - V3HandshakeCycles : std::max(Request.Earliest, Bus);
  // A tile store has no writeback to a backend: it is done when a scalar
  // store would be.
  Result.Done = Bus + Latency - 1 +
                (Request.IsTile && Request.IsStore ? V3ScalarStoreCycles
                                                   : V3ReturnCycles);
  if (Request.Earliest >= V3BackendLead)
    Result.Done = std::max(Result.Done, Request.Earliest - V3BackendLead);
  Result.Done = std::max(Result.Done, Request.ReceiverReady);
  // The slot is the instruction's from its first register until its last
  // one is done; a later register extends the booking.
  Cycle From = Pipe.getFreeAt(V3Slot);
  if (!Continues)
    From = std::max(From, Waits ? Bus - V3HandshakeCycles : Bus);
  if (Result.Done + 1 > From)
    Pipe.reserveOn(V3Slot, From, Result.Done + 1 - From);
  if (Request.FirstRegister) {
    Cycle Now = getClock()->currentCycle();
    OutCreditsPort.send(1, std::max(Bus, Now) - Now);
  }
  return Result;
}
