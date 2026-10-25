//===- Dispatch.h - The dispatch unit ---------------------------*- C++ -*-===//
///
/// \file
/// The dispatch unit: takes instructions from the frontend's instruction
/// buffer (InstBufferInterface) and sends them, in program order, to the unit
/// that executes them (getExecTarget()) and to the Rob.
///
/// Super-coarse, deliberately simpler than the M3 RTL:
/// - up to `dispatch_width` per cycle, strictly in order: the first
///   instruction that cannot go blocks everything behind it;
/// - an instruction waits while a register it reads or writes is still being
///   produced (scoreboard: RAW and WAW hazards);
/// - an instruction waits while the Rob or its unit's queue is full;
/// - not modeled yet: Reef's special dispatch rules (a branch ends the group,
///   FP and CSR instructions dispatch alone from slot 0, load/store address
///   registers need a registered value, CSRs wait for an empty Rob).
///
/// Every cycle in which nothing dispatches is charged to exactly one stall
/// reason, so the stall counters add up to cycles - cycles_with_dispatch.
/// Separately, every dispatch slot (dispatch_width per cycle) is charged to
/// one top-down category (TopDownSlots, docs/topdown.md).
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_CORE_DISPATCH_H
#define REEF_PERF_CORE_DISPATCH_H

#include "common/Inst.h"
#include "common/Interfaces.h"

#include "sparta/events/UniqueEvent.hpp"
#include "sparta/ports/DataPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace reef_perf {

/// Why nothing dispatched in a cycle.
enum class StallReason : uint8_t {
  IbufEmpty, ///< Nothing to dispatch: fetch is behind.
  Raw,       ///< A source register is not ready yet.
  Waw,       ///< A destination register still has a write in flight.
  RobFull,   ///< No free Rob entry.
  LsuFull,   ///< LSU queue full.
  VecFull,   ///< Vector command queue full.
  MtxFull,   ///< Matrix command queue full.
  NumReasons ///< Number of reasons (not a reason).
};

/// \returns the name of \p Reason in statistics and JSON, e.g. "raw_hazard".
const char *getStallReasonName(StallReason Reason);

/// \returns the top-down name of the backend part \p Target stands for:
/// "core" (scalar execution), "memory" (the LSU and memory), "vector" or
/// "matrix".
const char *getBoundName(ExecTarget Target);

/// Where the dispatch slots went: each cycle has `dispatch_width` slots, and
/// each slot is charged to exactly one counter here (docs/topdown.md).
struct TopDownSlots {
  /// Number of FrontendStall values.
  static constexpr size_t NumFrontend =
      static_cast<size_t>(FrontendStall::NumStalls);
  /// Number of StallReason values.
  static constexpr size_t NumReasons =
      static_cast<size_t>(StallReason::NumReasons);

  uint32_t Width = 0; ///< Slots per cycle (dispatch_width).
  /// Slots in which an instruction dispatched. Top-down calls them Retiring
  /// because the instruction eventually retires: the model never runs the
  /// wrong path, so every dispatched instruction does.
  uint64_t Retiring = 0;
  /// Slots with nothing to dispatch, by FrontendStall.
  std::array<uint64_t, NumFrontend> Frontend{};
  /// Slots blocked behind an instruction, by the ExecTarget responsible and
  /// the StallReason.
  std::array<std::array<uint64_t, NumReasons>, NumExecTargets> Backend{};

  /// \returns all slots counted so far.
  uint64_t getTotal() const;
};

class Rob;

/// The dispatch unit. Tree location: top.core.dispatch.
class Dispatch : public sparta::Unit {
public:
  // Parameter names are config keys, so they keep Sparta's snake_case.
  // NOLINTBEGIN(readability-identifier-naming)
  struct DispatchParameterSet : public sparta::ParameterSet {
    explicit DispatchParameterSet(sparta::TreeNode *Node)
        : sparta::ParameterSet(Node) {}
    PARAMETER(uint32_t, dispatch_width, 4,
              "Max instructions dispatched per cycle")
  };

  /// Sparta resource name; ResourceFactory requires this exact member name.
  static constexpr const char *name = "dispatch";
  // NOLINTEND(readability-identifier-naming)

  /// Creates the unit on \p Node with parameters \p Params.
  Dispatch(sparta::TreeNode *Node, const DispatchParameterSet *Params);

  /// Connects the frontend's instruction buffer \p Buffer, which must outlive
  /// the unit.
  void setInstBuffer(InstBufferInterface *Buffer) { Frontend = Buffer; }

  /// Connects the Rob \p R, so a full Rob can be blamed on what its oldest
  /// instruction waits for.
  void setRob(const Rob *R) { TheRob = R; }

  /// \returns where the dispatch slots went.
  const TopDownSlots &getTopDownSlots() const { return Slots; }

  /// \returns the cycles in which at least one instruction dispatched.
  uint64_t getCyclesWithDispatch() const { return CyclesWithDispatch.get(); }

  /// \returns the cycles in which nothing dispatched, blamed on \p Reason.
  uint64_t getStallCycles(StallReason Reason) const {
    return StallCounters.at(static_cast<size_t>(Reason))->get();
  }

private:
  // Sparta port handlers take `const DataT &`.
  void receiveRobCredits(const uint32_t &Credits);
  void receiveLsuCredits(const uint32_t &Credits);
  void receiveVecCredits(const uint32_t &Credits);
  void receiveMtxCredits(const uint32_t &Credits);

  void dispatchGroup();
  void scheduleDispatch();
  /// \returns true if \p I can dispatch at \p Now; otherwise sets \p Why and
  /// \p Blamed, the unit whose work holds \p I up.
  bool canDispatch(const InstRef &I, Cycle Now, StallReason &Why,
                   ExecTarget &Blamed) const;
  /// Sends \p I to the unit that executes it and spends that unit's credit.
  void sendToTarget(const InstRef &I);

  const uint32_t DispatchWidth;

  InstBufferInterface *Frontend = nullptr;
  const Rob *TheRob = nullptr;
  TopDownSlots Slots;
  // Scoreboard: the in-flight instruction that last wrote each register.
  std::array<InstRef, RegId::NumDense> LastWriter{};

  uint32_t RobCredits = 0;
  uint32_t LsuCredits = 0;
  uint32_t VecCredits = 0;
  uint32_t MtxCredits = 0;

  sparta::DataOutPort<InstRef> OutScalarPort{&unit_port_set_, "out_scalar"};
  sparta::DataOutPort<InstRef> OutLsuPort{&unit_port_set_, "out_lsu"};
  sparta::DataOutPort<InstRef> OutVectorPort{&unit_port_set_, "out_vector"};
  sparta::DataOutPort<InstRef> OutMatrixPort{&unit_port_set_, "out_matrix"};
  sparta::DataOutPort<InstRef> OutRobPort{&unit_port_set_, "out_rob"};
  sparta::DataInPort<uint32_t> InRobCreditsPort{&unit_port_set_,
                                                "in_rob_credits", 1};
  sparta::DataInPort<uint32_t> InLsuCreditsPort{&unit_port_set_,
                                                "in_lsu_credits", 1};
  sparta::DataInPort<uint32_t> InVecCreditsPort{&unit_port_set_,
                                                "in_vec_credits", 1};
  sparta::DataInPort<uint32_t> InMtxCreditsPort{&unit_port_set_,
                                                "in_mtx_credits", 1};

  sparta::UniqueEvent<> DispatchEvent{
      &unit_event_set_, "ev_dispatch",
      CREATE_SPARTA_HANDLER(Dispatch, dispatchGroup)};

  sparta::Counter NumDispatched{&unit_stat_set_, "num_dispatched",
                                "Instructions dispatched",
                                sparta::Counter::COUNT_NORMAL};
  sparta::Counter CyclesWithDispatch{
      &unit_stat_set_, "cycles_with_dispatch",
      "Cycles in which at least one instruction dispatched",
      sparta::Counter::COUNT_NORMAL};
  std::vector<std::unique_ptr<sparta::Counter>> StallCounters;
};

} // namespace reef_perf

#endif // REEF_PERF_CORE_DISPATCH_H
