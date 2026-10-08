#include "reef_perf/frontend/inst_decode.hpp"

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <vector>

namespace reef_perf {

namespace {

// Major opcodes (instruction bits 6:0).
constexpr std::uint32_t kOpLui = 0x37;
constexpr std::uint32_t kOpAuipc = 0x17;
constexpr std::uint32_t kOpJal = 0x6f;
constexpr std::uint32_t kOpJalr = 0x67;
constexpr std::uint32_t kOpBranch = 0x63;
constexpr std::uint32_t kOpLoad = 0x03;
constexpr std::uint32_t kOpStore = 0x23;
constexpr std::uint32_t kOpImm = 0x13;
constexpr std::uint32_t kOpReg = 0x33;
constexpr std::uint32_t kOpMiscMem = 0x0f;
constexpr std::uint32_t kOpSystem = 0x73;
constexpr std::uint32_t kOpLoadFp = 0x07;
constexpr std::uint32_t kOpStoreFp = 0x27;
constexpr std::uint32_t kOpFmadd = 0x43;
constexpr std::uint32_t kOpFmsub = 0x47;
constexpr std::uint32_t kOpFnmsub = 0x4b;
constexpr std::uint32_t kOpFnmadd = 0x4f;
constexpr std::uint32_t kOpFp = 0x53;
constexpr std::uint32_t kOpVector = 0x57;

// funct7 of the M extension.
constexpr std::uint32_t kFunct7MulDiv = 0x01;
// funct6 of the vector unary groups (VWXUNARY0, VRXUNARY0, ...).
constexpr std::uint32_t kFunct6Unary = 0x10;
// vsetvl encodes 1000000 in bits 31:25.
constexpr std::uint32_t kVsetvlTop7 = 0x40;

/** Extracts bits hi..lo (inclusive) of a word.
 *
 *  @param word Source word.
 *  @param hi Highest bit index.
 *  @param lo Lowest bit index.
 *  @return The field, right-aligned.
 */
std::uint32_t bits(std::uint32_t word, int hi, int lo) {
    return (word >> lo) & ((1U << (hi - lo + 1)) - 1U);
}

/** Whether a string starts with a prefix.
 *
 *  @param text String to test.
 *  @param prefix Expected prefix.
 *  @return True if text begins with prefix.
 */
bool starts_with(std::string_view text, std::string_view prefix) {
    return text.starts_with(prefix);
}

/** Adds an integer register, skipping x0.
 *
 *  @param regs List to append to.
 *  @param reg Register number 0..31.
 */
void add_x(std::vector<std::uint16_t>& regs, std::uint32_t reg) {
    if (reg != 0) {
        regs.push_back(static_cast<std::uint16_t>(kXBase + reg));
    }
}

/** Adds an FP register.
 *
 *  @param regs List to append to.
 *  @param reg Register number 0..31.
 */
void add_f(std::vector<std::uint16_t>& regs, std::uint32_t reg) {
    regs.push_back(static_cast<std::uint16_t>(kFBase + reg));
}

/** Adds a group of consecutive vector registers.
 *
 *  @param regs List to append to.
 *  @param base First register number.
 *  @param count Number of registers in the group.
 */
void add_v(std::vector<std::uint16_t>& regs, std::uint32_t base,
           std::uint32_t count) {
    for (std::uint32_t k = 0; k < count && base + k < 32; ++k) {
        regs.push_back(static_cast<std::uint16_t>(kVBase + base + k));
    }
}

/** Class of an OP-V arithmetic instruction, from its mnemonic.
 *
 *  @param mnemonic Instruction mnemonic.
 *  @return The vector class.
 */
InstClass vector_arith_class(std::string_view mnemonic) {
    const auto any_of = [mnemonic](std::initializer_list<std::string_view> ps) {
        return std::ranges::any_of(ps, [mnemonic](auto prefix) {
            return starts_with(mnemonic, prefix);
        });
    };
    if (any_of({"vmv.x.s", "vcpop", "vfirst", "vfmv.f.s"})) {
        return InstClass::V_TO_SCALAR;
    }
    if (any_of({"vred", "vwred", "vfred", "vfwred", "vrgather", "vslide",
                "vfslide", "vcompress"})) {
        return InstClass::V_PERM;
    }
    if (any_of({"vdiv", "vrem"})) {
        return InstClass::V_DIV;
    }
    if (any_of({"vfdiv", "vfrdiv", "vfsqrt"})) {
        return InstClass::V_FDIV;
    }
    if (any_of({"vmul", "vmacc", "vnmsac", "vmadd", "vnmsub", "vwmul", "vwmacc",
                "vsmul"})) {
        return InstClass::V_MUL;
    }
    if (starts_with(mnemonic, "vf")) {
        return InstClass::V_FP;
    }
    return InstClass::V_ALU;
}

/** Decodes the sources of an OP-V arithmetic instruction.
 *
 *  @param inst Instruction being decoded.
 *  @param word Raw encoding.
 *  @param emul Registers per operand group.
 *  @param vs2_regs Registers in the vs2 group (doubled when narrowing).
 */
void decode_vector_sources(Inst& inst, std::uint32_t word, std::uint32_t emul,
                           std::uint32_t vs2_regs) {
    const std::uint32_t funct3 = bits(word, 14, 12);
    const std::uint32_t funct6 = bits(word, 31, 26);
    const std::uint32_t vs2 = bits(word, 24, 20);
    const std::uint32_t rs1 = bits(word, 19, 15);
    const bool unary_wx = funct3 == 2 && funct6 == kFunct6Unary;
    const bool unary_wf = funct3 == 1 && funct6 == kFunct6Unary;
    const bool scalar_to_vec = (funct3 == 6 || funct3 == 5) &&
                               funct6 == kFunct6Unary; // vmv.s.x, vfmv.s.f
    switch (funct3) {
    case 0: // OPIVV
    case 1: // OPFVV
    case 2: // OPMVV
        if (!unary_wx && !unary_wf) {
            add_v(inst.srcs, rs1, emul);
        }
        add_v(inst.srcs, vs2, vs2_regs);
        break;
    case 3: // OPIVI
        add_v(inst.srcs, vs2, vs2_regs);
        break;
    case 4: // OPIVX
    case 6: // OPMVX
        add_x(inst.srcs, rs1);
        if (!scalar_to_vec) {
            add_v(inst.srcs, vs2, vs2_regs);
        }
        break;
    case 5: // OPFVF
        add_f(inst.srcs, rs1);
        if (!scalar_to_vec) {
            add_v(inst.srcs, vs2, vs2_regs);
        }
        break;
    default:
        break;
    }
}

/** Decodes an OP-V instruction (major opcode 0x57).
 *
 *  @param inst Instruction being decoded.
 *  @param word Raw encoding.
 */
void decode_op_v(Inst& inst, std::uint32_t word) {
    const std::uint32_t funct3 = bits(word, 14, 12);
    const std::uint32_t funct6 = bits(word, 31, 26);
    const std::uint32_t vm = bits(word, 25, 25);
    const std::uint32_t vs2 = bits(word, 24, 20);
    const std::uint32_t rs1 = bits(word, 19, 15);
    const std::uint32_t rd = bits(word, 11, 7);
    const std::string_view mnemonic = inst.mnemonic;

    if (funct3 == 7) { // vsetvli / vsetivli / vsetvl
        inst.cls = InstClass::VSET;
        if (bits(word, 31, 30) != 3) { // not vsetivli
            add_x(inst.srcs, rs1);
        }
        if (bits(word, 31, 25) == kVsetvlTop7) {
            add_x(inst.srcs, vs2);
        }
        add_x(inst.dsts, rd);
        return;
    }

    inst.cls = vector_arith_class(mnemonic);

    // Coarse: every operand spans LMUL registers; widening results span
    // 2*LMUL. Mask-producing ops are not special-cased.
    const std::uint32_t emul = std::max<std::uint32_t>(1, inst.lmul8 / 8);
    const bool widening =
        starts_with(mnemonic, "vw") || starts_with(mnemonic, "vfw");
    const bool narrowing =
        starts_with(mnemonic, "vn") || starts_with(mnemonic, "vfn");
    const std::uint32_t dst_regs = widening ? 2 * emul : emul;
    const std::uint32_t vs2_regs = narrowing ? 2 * emul : emul;

    decode_vector_sources(inst, word, emul, vs2_regs);
    if (vm == 0) {
        add_v(inst.srcs, 0, 1); // masked: reads v0
    }

    // Multiply-accumulate forms also read the destination.
    const bool int_mac = (funct3 == 2 || funct3 == 6) &&
                         (funct6 == 0x29 || funct6 == 0x2b || funct6 == 0x2d ||
                          funct6 == 0x2f || funct6 >= 0x3c);
    const bool fp_mac = (funct3 == 1 || funct3 == 5) &&
                        ((funct6 >= 0x28 && funct6 <= 0x2f) || funct6 >= 0x3c);
    if (int_mac || fp_mac) {
        add_v(inst.srcs, rd, dst_regs);
    }

    if (funct3 == 2 && funct6 == kFunct6Unary) { // vmv.x.s, vcpop, vfirst
        add_x(inst.dsts, rd);
    } else if (funct3 == 1 && funct6 == kFunct6Unary) { // vfmv.f.s
        add_f(inst.dsts, rd);
    } else {
        add_v(inst.dsts, rd, dst_regs);
    }
}

/** Decodes LOAD-FP / STORE-FP: scalar FP or vector memory ops.
 *
 *  @param inst Instruction being decoded.
 *  @param word Raw encoding.
 *  @param is_store True for STORE-FP.
 */
void decode_fp_or_vector_mem(Inst& inst, std::uint32_t word, bool is_store) {
    const std::uint32_t width = bits(word, 14, 12);
    const std::uint32_t rs1 = bits(word, 19, 15);
    const std::uint32_t rs2 = bits(word, 24, 20);
    const std::uint32_t rd = bits(word, 11, 7);
    if (width >= 1 && width <= 4) { // flh/flw/fld/flq (Reef has flw/fsw)
        inst.cls = is_store ? InstClass::FP_STORE : InstClass::FP_LOAD;
        add_x(inst.srcs, rs1);
        if (is_store) {
            add_f(inst.srcs, rs2);
        } else {
            add_f(inst.dsts, rd);
        }
        return;
    }
    inst.cls = is_store ? InstClass::V_STORE : InstClass::V_LOAD;
    const std::uint32_t mop = bits(word, 27, 26);
    const std::uint32_t nf = bits(word, 31, 29) + 1;
    const std::uint32_t emul = std::max<std::uint32_t>(1, inst.lmul8 / 8) * nf;
    add_x(inst.srcs, rs1);
    if (mop == 2) { // strided
        add_x(inst.srcs, rs2);
    }
    if (mop == 1 || mop == 3) { // indexed (coarse: one index register)
        add_v(inst.srcs, rs2, 1);
    }
    if (bits(word, 25, 25) == 0) { // masked
        add_v(inst.srcs, 0, 1);
    }
    if (is_store) {
        add_v(inst.srcs, rd, emul);
    } else {
        add_v(inst.dsts, rd, emul);
    }
}

/** Decodes an OP-FP instruction (major opcode 0x53).
 *
 *  @param inst Instruction being decoded.
 *  @param word Raw encoding.
 */
void decode_op_fp(Inst& inst, std::uint32_t word) {
    const std::uint32_t funct5 = bits(word, 31, 27);
    const std::uint32_t rs1 = bits(word, 19, 15);
    const std::uint32_t rs2 = bits(word, 24, 20);
    const std::uint32_t rd = bits(word, 11, 7);
    inst.cls = InstClass::FP;
    switch (funct5) {
    case 0x03: // fdiv.s
        inst.cls = InstClass::FP_DIV;
        add_f(inst.srcs, rs1);
        add_f(inst.srcs, rs2);
        add_f(inst.dsts, rd);
        break;
    case 0x0b: // fsqrt.s
        inst.cls = InstClass::FP_DIV;
        add_f(inst.srcs, rs1);
        add_f(inst.dsts, rd);
        break;
    case 0x14: // feq/flt/fle -> x
        add_f(inst.srcs, rs1);
        add_f(inst.srcs, rs2);
        add_x(inst.dsts, rd);
        break;
    case 0x18: // fcvt.w[u].s -> x
    case 0x1c: // fmv.x.w, fclass.s -> x
        add_f(inst.srcs, rs1);
        add_x(inst.dsts, rd);
        break;
    case 0x1a: // fcvt.s.w[u] <- x
    case 0x1e: // fmv.w.x <- x
        add_x(inst.srcs, rs1);
        add_f(inst.dsts, rd);
        break;
    case 0x08: // unary conversions between FP formats
        add_f(inst.srcs, rs1);
        add_f(inst.dsts, rd);
        break;
    default: // fadd, fsub, fmul, fsgnj*, fmin/fmax
        add_f(inst.srcs, rs1);
        add_f(inst.srcs, rs2);
        add_f(inst.dsts, rd);
        break;
    }
}

/** Decodes the scalar integer, control-flow and system opcodes.
 *
 *  @param inst Instruction being decoded.
 *  @param word Raw encoding.
 *  @return False if the opcode is not one of these.
 */
bool decode_scalar(Inst& inst, std::uint32_t word) {
    const std::uint32_t opcode = bits(word, 6, 0);
    const std::uint32_t rd = bits(word, 11, 7);
    const std::uint32_t funct3 = bits(word, 14, 12);
    const std::uint32_t rs1 = bits(word, 19, 15);
    const std::uint32_t rs2 = bits(word, 24, 20);
    switch (opcode) {
    case kOpLui:
    case kOpAuipc:
        inst.cls = InstClass::ALU;
        add_x(inst.dsts, rd);
        return true;
    case kOpJal:
        inst.cls = InstClass::JUMP;
        add_x(inst.dsts, rd);
        return true;
    case kOpJalr:
        inst.cls = InstClass::JUMP;
        add_x(inst.srcs, rs1);
        add_x(inst.dsts, rd);
        return true;
    case kOpBranch:
        inst.cls = InstClass::BRANCH;
        add_x(inst.srcs, rs1);
        add_x(inst.srcs, rs2);
        return true;
    case kOpLoad:
        inst.cls = InstClass::LOAD;
        add_x(inst.srcs, rs1);
        add_x(inst.dsts, rd);
        return true;
    case kOpStore:
        inst.cls = InstClass::STORE;
        add_x(inst.srcs, rs1);
        add_x(inst.srcs, rs2);
        return true;
    case kOpImm:
        inst.cls = InstClass::ALU;
        add_x(inst.srcs, rs1);
        add_x(inst.dsts, rd);
        return true;
    case kOpReg:
        if (bits(word, 31, 25) == kFunct7MulDiv) {
            inst.cls = funct3 < 4 ? InstClass::MUL : InstClass::DIV;
        } else {
            inst.cls = InstClass::ALU;
        }
        add_x(inst.srcs, rs1);
        add_x(inst.srcs, rs2);
        add_x(inst.dsts, rd);
        return true;
    case kOpMiscMem:
        inst.cls = InstClass::FENCE;
        return true;
    case kOpSystem:
        if (funct3 == 0) {
            inst.cls = InstClass::SYSTEM; // ecall, ebreak, mret, wfi, mpause
        } else {
            inst.cls = InstClass::CSR;
            if (funct3 <= 3) { // register (not immediate) forms
                add_x(inst.srcs, rs1);
            }
            add_x(inst.dsts, rd);
        }
        return true;
    default:
        return false;
    }
}

} // namespace

void decode_inst(Inst& inst) {
    // Mnemonic = first whitespace-separated token of the disassembly.
    const std::string& text = inst.disasm;
    inst.mnemonic = text.substr(0, text.find_first_of(" \t"));

    const std::uint32_t word = inst.encoding;
    inst.srcs.clear();
    inst.dsts.clear();
    inst.cls = InstClass::UNKNOWN;

    if (decode_scalar(inst, word)) {
        return;
    }
    switch (bits(word, 6, 0)) {
    case kOpLoadFp:
        decode_fp_or_vector_mem(inst, word, /*is_store=*/false);
        break;
    case kOpStoreFp:
        decode_fp_or_vector_mem(inst, word, /*is_store=*/true);
        break;
    case kOpFmadd:
    case kOpFmsub:
    case kOpFnmsub:
    case kOpFnmadd:
        inst.cls = InstClass::FP;
        add_f(inst.srcs, bits(word, 19, 15));
        add_f(inst.srcs, bits(word, 24, 20));
        add_f(inst.srcs, bits(word, 31, 27));
        add_f(inst.dsts, bits(word, 11, 7));
        break;
    case kOpFp:
        decode_op_fp(inst, word);
        break;
    case kOpVector:
        decode_op_v(inst, word);
        break;
    default:
        break;
    }
}

} // namespace reef_perf
