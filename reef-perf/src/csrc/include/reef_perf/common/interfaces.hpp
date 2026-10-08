#pragma once

/** @file
 *  @brief Types shared across module boundaries.
 *
 *  Everything one module needs to know about another lives here or in
 *  inst.hpp; see docs/interfaces.md for the ports that carry these types and
 *  for which module writes which Inst field.
 */

#include "reef_perf/common/inst.hpp"

#include <cstdint>
#include <span>

namespace reef_perf {

/// Who is asking the memory module for an access.
enum class Requester : std::uint8_t {
    IFETCH, ///< Instruction fetch (not routed through memory yet).
    LSU,    ///< The backend's load/store unit.
    MATRIX, ///< The matrix engine (not routed through memory yet).
};

/// One memory operation, as a requester hands it to the memory module.
struct MemRequest {
    /// Who is asking.
    Requester requester = Requester::LSU;
    /// The bytes accessed, e.g. one entry per vector element. Empty means a
    /// single access of unknown address; it is timed as one DTCM line.
    std::span<const MemAccess> accesses;
    /// First cycle the operation may start (the requester's own port is free
    /// from this cycle on).
    std::uint64_t earliest = 0;
};

/// The memory module's answer to a MemRequest.
struct MemResponse {
    /// Cycle the transfer starts; at least MemRequest::earliest.
    std::uint64_t start = 0;
    /// Cycles the requester's port is busy, from start.
    std::uint64_t occupancy = 1;
    /// Cycles from start until the data can be used (load) or the write is
    /// done (store). A dependent instruction may dispatch at
    /// start + latency - 1, following the scoreboard convention.
    std::uint64_t latency = 1;
};

/** The memory module as other modules see it.
 *
 *  Phase 1 is synchronous: access() books the memory's resources and returns
 *  the timing straight away, the same "book a slot" style as the execution
 *  pools. The memory module's owner may later replace it with request and
 *  response ports.
 */
class MemoryInterface {
  public:
    /// Requesters never own the memory module.
    virtual ~MemoryInterface() = default;

    /** Times one memory operation and books the resources it uses.
     *
     *  @param request The operation.
     *  @return When it starts, how long it holds the requester's port and
     *          its latency.
     */
    virtual MemResponse access(const MemRequest& request) = 0;
};

/// The unit that executes an instruction, and so owns its timing fields
/// (issue_cycle, result_ready_cycle, complete_cycle).
enum class ExecTarget : std::uint8_t {
    SCALAR, ///< backend.scalar_exec: integer, FP, branch, CSR, system.
    LSU,    ///< backend.lsu: scalar, FP and vector loads and stores.
    VECTOR, ///< vector.vxu: vector arithmetic and vset*.
};

/** Where Dispatch sends an instruction.
 *
 *  Memory instructions, including vector ones, go to the LSU: on M3 one LSU
 *  serves both the scalar core and the RVV backend.
 *
 *  @param inst A decoded instruction.
 *  @return The unit that executes it.
 */
inline ExecTarget exec_target(const Inst& inst) {
    if (inst.is_memory()) {
        return ExecTarget::LSU;
    }
    if (inst.is_vector()) {
        return ExecTarget::VECTOR;
    }
    return ExecTarget::SCALAR;
}

} // namespace reef_perf
