//===- Interfaces.h - Types shared across modules ---------------*- C++ -*-===//
///
/// \file
/// Types shared across module boundaries. The ports that carry them, and
/// which module writes which Inst field, are listed in docs/interfaces.md.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_COMMON_INTERFACES_H
#define REEF_PERF_COMMON_INTERFACES_H

#include "common/Inst.h"
#include "common/Types.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace reef_perf {

/// Who is asking the memory module for an access.
enum class RequesterKind : uint8_t {
  IFetch, ///< Instruction fetch (not routed through memory yet).
  LSU,    ///< The core's load/store unit.
  Matrix  ///< The matrix engine (not routed through memory yet).
};

/// One memory operation, as a requester hands it to the memory module.
struct MemRequest {
  RequesterKind Requester = RequesterKind::LSU; ///< Who is asking.
  /// The bytes accessed. Empty means one access of unknown address, timed as
  /// one DTCM line.
  std::span<const MemAccess> Accesses;
  /// First cycle the operation may start (the requester's port is free).
  Cycle Earliest = 0;
};

/// The memory module's answer to a MemRequest.
struct MemResponse {
  Cycle Start = 0;        ///< Cycle the transfer starts (>= Earliest).
  uint64_t Occupancy = 1; ///< Cycles the requester's port stays busy.
  /// Cycles from Start until the data can be used or the write is done.
  uint64_t Latency = 1;
};

/// The memory module as other modules see it. Synchronous for now: access()
/// books the memory's resources and returns the timing at once ("book a
/// slot", like the execution pools). The memory owner may replace it with
/// request/response ports later.
class MemoryInterface {
public:
  virtual ~MemoryInterface() = default; ///< Requesters never own memory.

  /// Times \p Request and books the resources it uses. \returns its timing.
  virtual MemResponse access(const MemRequest &Request) = 0;
};

/// Why the frontend had no instruction for the core in a cycle, for top-down
/// accounting (docs/topdown.md).
enum class FrontendStall : uint8_t {
  BranchResteer,  ///< Refetching after a taken branch, jump or trap.
  FetchBandwidth, ///< Fetch hasn't caught up (width, interval, latency).
  NumStalls       ///< Number of reasons (not a reason).
};

/// \returns the name of \p Stall in statistics and JSON, e.g.
/// "branch_resteer".
inline const char *getFrontendStallName(FrontendStall Stall) {
  switch (Stall) {
  case FrontendStall::BranchResteer:
    return "branch_resteer";
  case FrontendStall::FetchBandwidth:
    return "fetch_bandwidth";
  case FrontendStall::NumStalls:
    break;
  }
  return "unknown";
}

/// The frontend's instruction buffer as the core sees it. Synchronous,
/// like MemoryInterface: the core looks at the oldest instruction and
/// takes it. To match a one-cycle hand-off in each direction, an instruction
/// becomes visible the cycle after it is fetched, and an entry the core
/// frees can be refilled from the next cycle.
class InstBufferInterface {
public:
  virtual ~InstBufferInterface() = default; ///< The core never owns it.

  /// \returns the oldest instruction the core may take at cycle \p Now, or
  /// nullptr if there is none yet.
  virtual InstRef peek(Cycle Now) const = 0;

  /// Removes the instruction peek(\p Now) returned; the core took it.
  virtual void pop(Cycle Now) = 0;

  /// \returns true once the program is fully fetched, as seen at cycle \p Now,
  /// and every instruction has been taken.
  virtual bool isDrained(Cycle Now) const = 0;

  /// \returns why peek(\p Now) found nothing.
  virtual FrontendStall getStall(Cycle Now) const = 0;
};

/// The unit that executes an instruction, and so writes its issue,
/// result-ready and complete cycles.
enum class ExecTarget : uint8_t {
  Scalar, ///< core.scalar_exec: integer, FP, branch, CSR, system.
  LSU,    ///< core.lsu: scalar, FP and vector loads and stores.
  Vector, ///< vector: vector arithmetic and vset*.
  Matrix  ///< matrix: matrix-engine ops.
};

/// Number of ExecTarget values.
constexpr size_t NumExecTargets = 4;

/// \returns the unit that executes \p I. Vector memory ops go to the LSU: on
/// M3 one LSU serves both the scalar core and the RVV backend.
inline ExecTarget getExecTarget(const Inst &I) {
  if (I.isMemory())
    return ExecTarget::LSU;
  if (I.isMatrix())
    return ExecTarget::Matrix;
  if (I.isVector())
    return ExecTarget::Vector;
  return ExecTarget::Scalar;
}

} // namespace reef_perf

#endif // REEF_PERF_COMMON_INTERFACES_H
