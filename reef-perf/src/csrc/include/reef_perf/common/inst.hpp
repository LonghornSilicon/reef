#pragma once

/** @file
 *  @brief The instruction object that flows through the timing model.
 *
 *  One Inst is created per dynamic instruction, at fetch, from the record the
 *  functional simulator produced when it executed the instruction
 *  (execute-at-fetch). Units pass InstPtr over Sparta ports and fill in the
 *  timing fields as the instruction moves down the pipeline.
 */

#include <cstdint>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace reef_perf {

/** Coarse instruction classes.
 *
 *  Each class maps to one execution resource in Execute and to a latency and
 *  occupancy parameter.
 */
enum class InstClass : std::uint8_t {
    ALU,         ///< Integer ALU, Zbb, lui/auipc.
    BRANCH,      ///< Conditional branches.
    JUMP,        ///< jal, jalr.
    MUL,         ///< mul*.
    DIV,         ///< div*, rem*.
    CSR,         ///< csrr*.
    FENCE,       ///< fence, fence.i.
    SYSTEM,      ///< ecall, ebreak, mret, wfi, mpause.
    LOAD,        ///< Scalar integer loads.
    STORE,       ///< Scalar integer stores.
    FP,          ///< FP32 add/mul/fma/compare/convert/move.
    FP_DIV,      ///< fdiv.s, fsqrt.s.
    FP_LOAD,     ///< flw.
    FP_STORE,    ///< fsw.
    VSET,        ///< vsetvli, vsetivli, vsetvl.
    V_ALU,       ///< Vector integer ALU, mask ops, moves.
    V_MUL,       ///< Vector multiply and multiply-accumulate.
    V_DIV,       ///< Vector integer divide.
    V_FP,        ///< Vector FP32 arithmetic.
    V_FDIV,      ///< Vector FP divide and square root.
    V_PERM,      ///< Reductions, slides, gathers, compress.
    V_LOAD,      ///< Vector loads.
    V_STORE,     ///< Vector stores.
    V_TO_SCALAR, ///< Vector ops that write a scalar or FP register.
    UNKNOWN,     ///< Not recognised; timed as an ALU op.
    NUM_CLASSES  ///< Number of classes (not a class).
};

/** Returns the short lower-case name of an instruction class.
 *
 *  @param cls The class to name.
 *  @return A static string such as "alu" or "v_load".
 */
const char* class_name(InstClass cls);

/// First scoreboard id of the integer registers x0..x31.
constexpr std::uint16_t kXBase = 0;
/// First scoreboard id of the FP registers f0..f31.
constexpr std::uint16_t kFBase = 32;
/// First scoreboard id of the vector registers v0..v31.
constexpr std::uint16_t kVBase = 64;
/// Number of scoreboard ids (all three register files).
constexpr std::uint16_t kNumRegs = 96;

/// One memory access performed by an instruction.
struct MemAccess {
    /// Byte address of the access.
    std::uint32_t addr = 0;
    /// Size of the access in bytes.
    std::uint8_t size = 0;
    /// True for a store, false for a load.
    bool is_store = false;
};

/// One dynamic instruction and its progress through the timing model.
struct Inst {
    /// Marks a timing field that has not been set yet.
    static constexpr std::uint64_t kNever =
        std::numeric_limits<std::uint64_t>::max();

    /// 0-based dynamic instruction number.
    std::uint64_t seq = 0;
    /// Address of the instruction.
    std::uint32_t pc = 0;
    /// Address of the next instruction executed (the branch outcome).
    std::uint32_t next_pc = 0;
    /// Raw 32-bit instruction word.
    std::uint32_t encoding = 0;
    /// Disassembly from the functional simulator.
    std::string disasm;
    /// First token of the disassembly, e.g. "addi".
    std::string mnemonic;
    /// Memory accesses, one per element for vector memory ops.
    std::vector<MemAccess> mem;
    /// Vector length in effect when the instruction executed.
    std::uint32_t vl = 0;
    /// Selected element width in bytes.
    std::uint8_t sew_bytes = 1;
    /// LMUL times eight (LMUL=1 is 8, LMUL=1/2 is 4).
    std::uint8_t lmul8 = 8;

    /// Coarse class, set by decode_inst().
    InstClass cls = InstClass::UNKNOWN;
    /// Registers read, as scoreboard ids (x0 excluded).
    std::vector<std::uint16_t> srcs;
    /// Registers written, as scoreboard ids (x0 excluded).
    std::vector<std::uint16_t> dsts;

    /// Cycle the instruction was fetched.
    std::uint64_t fetch_cycle = 0;
    /// Cycle the instruction was dispatched.
    std::uint64_t dispatch_cycle = 0;
    /// Cycle the instruction started in its execution resource.
    std::uint64_t issue_cycle = 0;
    /// First cycle a dependent instruction may dispatch.
    std::uint64_t result_ready_cycle = kNever;
    /// First cycle the instruction may retire.
    std::uint64_t complete_cycle = kNever;
    /// Cycle the instruction retired.
    std::uint64_t retire_cycle = 0;

    /** Whether the next instruction is not at pc + 4.
     *
     *  @return True for a taken branch, a jump or a trap.
     */
    [[nodiscard]] bool redirected() const { return next_pc != pc + 4; }

    /** Whether the instruction is handled by the vector unit or the LSU's
     *  vector path.
     *
     *  @return True for VSET through V_TO_SCALAR.
     */
    [[nodiscard]] bool is_vector() const {
        return cls >= InstClass::VSET && cls <= InstClass::V_TO_SCALAR;
    }

    /** Whether the instruction accesses memory through the LSU.
     *
     *  @return True for scalar, FP and vector loads and stores.
     */
    [[nodiscard]] bool is_memory() const {
        return cls == InstClass::LOAD || cls == InstClass::STORE ||
               cls == InstClass::FP_LOAD || cls == InstClass::FP_STORE ||
               cls == InstClass::V_LOAD || cls == InstClass::V_STORE;
    }
};

/// Shared handle to an instruction; this is what Sparta ports carry.
using InstPtr = std::shared_ptr<Inst>;

/// What Fetch sends to Dispatch in one fetch cycle.
struct FetchPacket {
    /// Instructions fetched this cycle, in program order.
    std::vector<InstPtr> insts;
    /// True once the program has halted; no more packets follow.
    bool last = false;
};

/** Prints an instruction for Sparta's port logging.
 *
 *  @param os Stream to print to.
 *  @param inst Instruction to print.
 *  @return The stream.
 */
std::ostream& operator<<(std::ostream& os, const Inst& inst);

/** Prints a fetch packet for Sparta's port logging.
 *
 *  @param os Stream to print to.
 *  @param pkt Packet to print.
 *  @return The stream.
 */
std::ostream& operator<<(std::ostream& os, const FetchPacket& pkt);

} // namespace reef_perf
