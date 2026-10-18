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

#include <cstdint>
#include <span>

namespace reef_perf {

/// Who is asking the memory module for an access.
enum class RequesterKind : uint8_t {
  IFetch, ///< Instruction fetch (not routed through memory yet).
  LSU,    ///< The backend's load/store unit.
  Matrix  ///< The matrix engine (not routed through memory yet).
};

/// One memory operation, as a requester hands it to the memory module.
struct MemRequest {
  RequesterKind Requester = RequesterKind::LSU; ///< Who is asking.
  /// The bytes accessed. Empty means one access of unknown address, timed as
  /// one DTCM line.
  std::span<const MemAccess> Accesses;
  /// First cycle the operation may start (the requester's port is free).
  uint64_t Earliest = 0;
};

/// The memory module's answer to a MemRequest.
struct MemResponse {
  uint64_t Start = 0;     ///< Cycle the transfer starts (>= Earliest).
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

/// The unit that executes an instruction, and so writes its issue,
/// result-ready and complete cycles.
enum class ExecTarget : uint8_t {
  Scalar, ///< backend.scalar_exec: integer, FP, branch, CSR, system.
  LSU,    ///< backend.lsu: scalar, FP and vector loads and stores.
  Vector, ///< vector: vector arithmetic and vset*.
  Matrix  ///< matrix: matrix-engine ops.
};

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
