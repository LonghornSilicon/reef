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

namespace reef_perf {

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
