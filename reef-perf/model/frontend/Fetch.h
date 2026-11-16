//===- Fetch.h - The frontend module ----------------------------*- C++ -*-===//
///
/// \file
/// The frontend module: fetch and the instruction buffer. Every fetched
/// instruction is executed in Spike first (FuncSim), so the frontend knows
/// each branch outcome. The core takes instructions from the buffer
/// through InstBufferInterface; on M3 the buffer is part of fetch
/// (UncachedFetch.scala, InstructionBuffer).
///
/// Super-coarse, deliberately simpler than the M3 RTL:
/// - up to `fetch_width` instructions every `fetch_interval` cycles, while
///   the `ibuf_entries`-entry buffer has room;
/// - a group ends at any instruction that redirects the PC (taken branch,
///   jump, trap) or that the predictor predicts taken; the next group starts
///   `redirect_penalty` cycles later if the prediction was wrong. `predictor`
///   is `none` (nothing predicted taken) or `btfn` (M3's predecode,
///   UncachedFetch.scala:87-100);
/// - with `fetch_reorder_buffer` on, the fetch unit CoralNPU has after M3
///   instead: a fetch is issued every cycle down the next line while fewer
///   than `fetch_max_in_flight` are outstanding; a line is written to the
///   buffer `fetch_response_cycles` later, in order, and only when all of it
///   fits. A predicted jump is seen when its line is written, and its target
///   is fetched in that cycle. A mispredicted one is fetched
///   `redirect_from_dispatch_cycles` after the core dispatches it;
/// - with `fetch_block_bytes` set, a group never crosses an aligned block of
///   that size: it ends at an instruction whose next PC is in another block
///   (M3: 16-byte lines, UncachedFetch.scala:56,107-131).
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_FRONTEND_FETCH_H
#define REEF_PERF_FRONTEND_FETCH_H

#include "common/Inst.h"
#include "common/Interfaces.h"
#include "frontend/FuncSim.h"

#include "sparta/events/UniqueEvent.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace reef_perf {

/// The frontend module. Tree location: top.frontend.
class Fetch : public sparta::Unit, public InstBufferInterface {
public:
  // Parameter names are config keys, so they keep Sparta's snake_case.
  // NOLINTBEGIN(readability-identifier-naming)
  struct FetchParameterSet : public sparta::ParameterSet {
    explicit FetchParameterSet(sparta::TreeNode *Node)
        : sparta::ParameterSet(Node) {}
    PARAMETER(uint32_t, fetch_width, 4,
              "Max instructions fetched per fetch cycle")
    PARAMETER(uint32_t, fetch_interval, 1,
              "Cycles between the starts of two fetch groups")
    PARAMETER(uint32_t, redirect_penalty, 1,
              "Extra cycles after a redirect the predictor did not predict")
    // CoralNPU's fetch unit after M3 (UncachedFetch.scala at 8a5e1b31:
    // FetchControl, Fetcher, FetchReorderBuffer). With fetch_reorder_buffer
    // off, the default and M3, the three values after it are not read; on,
    // fetch_interval and redirect_penalty are not.
    PARAMETER(bool, fetch_reorder_buffer, false,
              "Fetch a line every cycle, speculatively down the next line, "
              "through a reorder buffer that commits whole lines in order")
    PARAMETER(uint32_t, fetch_max_in_flight, 2,
              "Reorder buffer entries: fetches issued and not yet committed")
    PARAMETER(uint32_t, fetch_response_cycles, 2,
              "Cycles from issuing a fetch to the earliest cycle its line "
              "can be written to the instruction buffer")
    PARAMETER(uint32_t, redirect_from_dispatch_cycles, 2,
              "Cycles from the core dispatching a mispredicted branch or "
              "jump to the fetch of its real target")
    PARAMETER(std::string, predictor, "none",
              "Branch predictor: none (nothing predicted taken) or btfn "
              "(jal and backward branches predicted taken, as on M3)")
    PARAMETER(uint32_t, fetch_block_bytes, 0,
              "Aligned fetch block in bytes; a group never crosses one "
              "(0: no blocks)")
    PARAMETER(uint32_t, ibuf_entries, 8, "Instruction buffer entries")
    PARAMETER(bool, zvt_in_spike, false,
              "With zvt: Spike executes Zvt instructions, and reef-perf's "
              "own Zvt layer runs beside it for the matrix details and is "
              "checked against it. Needs reef_perf_pinned (a Spike with "
              "Zvt). Off: the layer executes them, as on reef_perf")
    PARAMETER(bool, decode_unary_vs1_is_opcode, false,
              "Decode the vs1 field of vzext/vsext, vmsbf/vmsif/vmsof, "
              "viota/vid, vfsqrt, vfcvt and vfclass as the sub-opcode it is, "
              "and give vzext/vsext their narrower source. Off: the M3 "
              "model, which reads it as a source register")
    PARAMETER(bool, zvt, false,
              "Run and decode Zvt matrix instructions, as CoralNPU builds "
              "with VME do. Off, a Zvt word traps, as on M3")
  };

  /// Sparta resource name; ResourceFactory requires this exact member name.
  static constexpr const char *name = "fetch";
  // NOLINTEND(readability-identifier-naming)
  /// Branch predictors fetch can model.
  enum class PredictorKind { None, Btfn };

  /// Creates the unit on \p Node with parameters \p Params.
  Fetch(sparta::TreeNode *Node, const FetchParameterSet *Params);

  /// Connects \p Sim, which must outlive the unit, and passes it the `zvt`
  /// parameter.
  void setFuncSim(FuncSim *Sim) {
    TheFuncSim = Sim;
    TheFuncSim->setZvt(Zvt);
    TheFuncSim->setUnaryVs1IsOpcode(UnaryVs1IsOpcode);
    TheFuncSim->setZvtInSpike(Zvt && ZvtInSpike);
  }

  /// Connects the memory module \p NewMem, where each fetch group books its
  /// ITCM line. Fetch keeps its own timing (fetch_interval); the booking
  /// shows other ITCM requesters (host, DMA) the cycles fetch uses.
  void setMemory(MemoryInterface *NewMem) { Mem = NewMem; }

  /// Holds the first fetch until cycle \p Cycles + 1, for a core that leaves
  /// reset after the host's boot load (memory/Host.h).
  void setStartCycle(Cycle Cycles) { StartCycle = Cycles; }

  /// \returns the taken branches, jumps and traps fetched so far.
  uint64_t getNumRedirects() const { return NumRedirects.get(); }
  /// \returns the oldest buffered instruction fetched before \p Now.
  InstRef peek(Cycle Now) const override;
  /// Removes the oldest instruction; its entry refills from \p Now + 1.
  void pop(Cycle Now) override;
  /// \returns true once fetch finished before \p Now and the buffer is empty.
  bool isDrained(Cycle Now) const override;
  /// \returns BranchResteer if the latest fetch group ended in a redirect,
  /// and FetchBandwidth otherwise.
  FrontendStall getStall(Cycle Now) const override;

private:
  void fetchGroup();
  void fetchSpeculative();
  void scheduleFirstFetch();
  void scheduleFetch(Cycle Earliest);
  /// \returns the entries fetch may fill at \p Now. Entries the core freed
  /// at \p Now only count from the next cycle.
  uint32_t getFreeEntries(Cycle Now) const;
  /// \returns true if the configured predictor predicts \p I taken.
  bool isPredictedTaken(const Inst &I) const;

  const uint32_t FetchWidth;
  const uint32_t FetchInterval;
  const bool ReorderBuffer;
  const uint32_t MaxInFlight;
  const uint32_t ResponseCycles;
  const uint32_t RedirectFromDispatch;
  const uint32_t RedirectPenalty;
  const PredictorKind Predictor;
  const uint32_t FetchBlockBytes;
  const uint32_t IbufEntries;
  const bool Zvt;
  const bool UnaryVs1IsOpcode;
  const bool ZvtInSpike;

  FuncSim *TheFuncSim = nullptr;
  MemoryInterface *Mem = nullptr;
  Cycle StartCycle = 0;
  std::deque<InstRef> Ibuf;  // Oldest first.
  Cycle FreedCycle = 0;      // Cycle of the latest pop()s ...
  uint32_t FreedAtCycle = 0; // ... and how many entries they freed.
  Cycle NextFetchCycle = 0;  // Earliest start of the next group.
  bool LastGroupRedirected = false;
  /// A fetched line in the reorder buffer.
  struct Line {
    Cycle Ready = 0;            ///< First cycle it can be committed.
    std::vector<InstRef> Insts; ///< Its instructions on the executed path.
    bool Redirects = false;     ///< Ends in a taken or predicted-taken jump.
    bool WrongPath = false;     ///< A sequential fetch behind a predicted jump.
  };
  std::deque<Line> InFlight;   // Issued, not yet committed; oldest first.
  Cycle NextIssue = 0;         // Earliest cycle of the next fetch.
  bool WaitsForCommit = false; // A predicted jump's line is in flight.
  InstRef WaitsForDispatch;    // A mispredicted jump the core has not taken.
  bool Exhausted = false;      // FuncSim has no more instructions.
  /// With the reorder buffer full behind a predicted jump, FetchControl
  /// holds a request for the next sequential line that it cannot withdraw
  /// (io.fetchAddr is Irrevocable, ongoingFetch).
  bool PendingWrongPath = false;
  bool Done = false;   // The program is fully fetched ...
  Cycle DoneCycle = 0; // ... as of this cycle.

  sparta::UniqueEvent<> FetchEvent{&unit_event_set_, "ev_fetch",
                                   CREATE_SPARTA_HANDLER(Fetch, fetchGroup)};

  sparta::Counter NumFetched{&unit_stat_set_, "num_fetched",
                             "Instructions fetched",
                             sparta::Counter::COUNT_NORMAL};
  sparta::Counter NumGroups{&unit_stat_set_, "num_groups",
                            "Fetch groups added to the instruction buffer",
                            sparta::Counter::COUNT_NORMAL};
  sparta::Counter NumRedirects{&unit_stat_set_, "num_redirects",
                               "Taken branches, jumps and traps",
                               sparta::Counter::COUNT_NORMAL};
  sparta::Counter NumMispredicts{
      &unit_stat_set_, "num_mispredicts",
      "Mispredicted branches/jumps and traps (each costs mispredict_penalty)",
      sparta::Counter::COUNT_NORMAL};
  sparta::Counter CyclesNoCredit{
      &unit_stat_set_, "cycles_no_credit",
      "Fetch cycles lost because the instruction buffer was full",
      sparta::Counter::COUNT_NORMAL};
};

} // namespace reef_perf

#endif // REEF_PERF_FRONTEND_FETCH_H
