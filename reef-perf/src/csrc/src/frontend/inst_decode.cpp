#include "reef_perf/frontend/inst_decode.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
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

/// Which register file an rs1/rs2/rs3 field reads.
enum class File : std::uint8_t {
    NONE, ///< The field is not a register source.
    X,    ///< Integer register.
    F,    ///< FP register.
};

/// Class and source register files of a scalar or FP instruction.
struct ScalarForm {
    InstClass cls = InstClass::UNKNOWN; ///< Instruction class.
    File rs1 = File::NONE;              ///< What bits 19:15 read.
    File rs2 = File::NONE;              ///< What bits 24:20 read.
    File rs3 = File::NONE;              ///< What bits 31:27 read.
};

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

/** Adds a scalar register source, skipping x0.
 *
 *  @param regs List to append to.
 *  @param file Register file; File::NONE adds nothing.
 *  @param reg Register number 0..31.
 */
void add(std::vector<std::uint16_t>& regs, File file, std::uint32_t reg) {
    if (file == File::X && reg != 0) {
        regs.push_back(static_cast<std::uint16_t>(kXBase + reg));
    } else if (file == File::F) {
        regs.push_back(static_cast<std::uint16_t>(kFBase + reg));
    }
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

/** Class and sources of an OP-FP instruction (major opcode 0x53).
 *
 *  @param funct5 Instruction bits 31:27.
 *  @return The form.
 */
ScalarForm op_fp_form(std::uint32_t funct5) {
    switch (funct5) {
    case 0x03: // fdiv.s
        return {.cls = InstClass::FP_DIV, .rs1 = File::F, .rs2 = File::F};
    case 0x0b: // fsqrt.s
        return {.cls = InstClass::FP_DIV, .rs1 = File::F};
    case 0x1a: // fcvt.s.w[u]
    case 0x1e: // fmv.w.x
        return {.cls = InstClass::FP, .rs1 = File::X};
    case 0x08: // conversions between FP formats
    case 0x18: // fcvt.w[u].s
    case 0x1c: // fmv.x.w, fclass.s
        return {.cls = InstClass::FP, .rs1 = File::F};
    default: // fadd, fsub, fmul, fsgnj*, fmin/fmax, feq/flt/fle
        return {.cls = InstClass::FP, .rs1 = File::F, .rs2 = File::F};
    }
}

/** Class and sources of every non-vector instruction.
 *
 *  @param word Raw encoding.
 *  @return The form, or std::nullopt for vector instructions and unknown
 *          opcodes.
 */
std::optional<ScalarForm> scalar_form(std::uint32_t word) {
    const std::uint32_t funct3 = bits(word, 14, 12);
    // LOAD-FP and STORE-FP widths 1-4 are scalar FP; the rest are vector.
    const bool scalar_fp_width = funct3 >= 1 && funct3 <= 4;
    switch (bits(word, 6, 0)) {
    case kOpLui:
    case kOpAuipc:
        return ScalarForm{.cls = InstClass::ALU};
    case kOpJal:
        return ScalarForm{.cls = InstClass::JUMP};
    case kOpJalr:
        return ScalarForm{.cls = InstClass::JUMP, .rs1 = File::X};
    case kOpBranch:
        return ScalarForm{
            .cls = InstClass::BRANCH, .rs1 = File::X, .rs2 = File::X};
    case kOpLoad:
        return ScalarForm{.cls = InstClass::LOAD, .rs1 = File::X};
    case kOpStore:
        return ScalarForm{
            .cls = InstClass::STORE, .rs1 = File::X, .rs2 = File::X};
    case kOpImm:
        return ScalarForm{.cls = InstClass::ALU, .rs1 = File::X};
    case kOpReg:
        if (bits(word, 31, 25) == kFunct7MulDiv) {
            return ScalarForm{.cls =
                                  funct3 < 4 ? InstClass::MUL : InstClass::DIV,
                              .rs1 = File::X,
                              .rs2 = File::X};
        }
        return ScalarForm{
            .cls = InstClass::ALU, .rs1 = File::X, .rs2 = File::X};
    case kOpMiscMem:
        return ScalarForm{.cls = InstClass::FENCE};
    case kOpSystem: // funct3 0: ecall, ebreak, mret, wfi, mpause
        if (funct3 == 0) {
            return ScalarForm{.cls = InstClass::SYSTEM};
        }
        // csrrw/s/c read rs1; the immediate forms (funct3 5-7) do not.
        return ScalarForm{.cls = InstClass::CSR,
                          .rs1 = funct3 <= 3 ? File::X : File::NONE};
    case kOpLoadFp:
        if (!scalar_fp_width) {
            return std::nullopt;
        }
        return ScalarForm{.cls = InstClass::FP_LOAD, .rs1 = File::X};
    case kOpStoreFp:
        if (!scalar_fp_width) {
            return std::nullopt;
        }
        return ScalarForm{
            .cls = InstClass::FP_STORE, .rs1 = File::X, .rs2 = File::F};
    case kOpFmadd:
    case kOpFmsub:
    case kOpFnmsub:
    case kOpFnmadd:
        return ScalarForm{.cls = InstClass::FP,
                          .rs1 = File::F,
                          .rs2 = File::F,
                          .rs3 = File::F};
    case kOpFp:
        return op_fp_form(bits(word, 31, 27));
    default:
        return std::nullopt;
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
            return mnemonic.starts_with(prefix);
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
    if (mnemonic.starts_with("vf")) {
        return InstClass::V_FP;
    }
    return InstClass::V_ALU;
}

/** Whether an OP-V instruction is a multiply-accumulate, which also reads
 *  its destination.
 *
 *  @param funct3 Instruction bits 14:12.
 *  @param funct6 Instruction bits 31:26.
 *  @return True for vmacc, vnmsac, vmadd, vnmsub, the widening integer
 *          forms and the FP fused multiply-adds.
 */
bool reads_destination(std::uint32_t funct3, std::uint32_t funct6) {
    const bool int_mac = (funct3 == 2 || funct3 == 6) &&
                         (funct6 == 0x29 || funct6 == 0x2b || funct6 == 0x2d ||
                          funct6 == 0x2f || funct6 >= 0x3c);
    const bool fp_mac = (funct3 == 1 || funct3 == 5) &&
                        ((funct6 >= 0x28 && funct6 <= 0x2f) || funct6 >= 0x3c);
    return int_mac || fp_mac;
}

/** Decodes an OP-V instruction (major opcode 0x57): class and sources.
 *
 *  Coarse: every operand spans LMUL registers, narrowing sources and
 *  widening accumulators span 2*LMUL.
 *
 *  @param inst Instruction being decoded.
 *  @param word Raw encoding.
 */
void decode_op_v(Inst& inst, std::uint32_t word) {
    const std::uint32_t funct3 = bits(word, 14, 12);
    const std::uint32_t funct6 = bits(word, 31, 26);
    const std::uint32_t vs2 = bits(word, 24, 20);
    const std::uint32_t rs1 = bits(word, 19, 15);
    const std::uint32_t rd = bits(word, 11, 7);
    const std::string_view mnemonic = inst.mnemonic;

    if (funct3 == 7) { // vsetvli / vsetivli / vsetvl
        inst.cls = InstClass::VSET;
        add(inst.srcs, bits(word, 31, 30) == 3 ? File::NONE : File::X, rs1);
        add(inst.srcs, bits(word, 31, 25) == kVsetvlTop7 ? File::X : File::NONE,
            vs2);
        return;
    }
    inst.cls = vector_arith_class(mnemonic);

    const std::uint32_t emul = std::max<std::uint32_t>(1, inst.lmul8 / 8);
    const bool widening =
        mnemonic.starts_with("vw") || mnemonic.starts_with("vfw");
    const bool narrowing =
        mnemonic.starts_with("vn") || mnemonic.starts_with("vfn");
    const std::uint32_t vs2_regs = narrowing ? 2 * emul : emul;
    const bool unary = funct6 == kFunct6Unary;

    // The first operand: vs1, or the scalar rs1, depending on funct3.
    switch (funct3) {
    case 0: // OPIVV
    case 1: // OPFVV
    case 2: // OPMVV
        // vmv.x.s, vfmv.f.s and the other unary ops have no vs1.
        if (!unary || funct3 == 0) {
            add_v(inst.srcs, rs1, emul);
        }
        break;
    case 4: // OPIVX
    case 6: // OPMVX
        add(inst.srcs, File::X, rs1);
        break;
    case 5: // OPFVF
        add(inst.srcs, File::F, rs1);
        break;
    default: // OPIVI: rs1 is an immediate
        break;
    }
    // vs2, except for scalar-to-vector moves (vmv.s.x, vfmv.s.f).
    if (!unary || (funct3 != 5 && funct3 != 6)) {
        add_v(inst.srcs, vs2, vs2_regs);
    }
    if (bits(word, 25, 25) == 0) { // masked: reads v0
        add_v(inst.srcs, 0, 1);
    }
    if (reads_destination(funct3, funct6)) {
        add_v(inst.srcs, rd, widening ? 2 * emul : emul);
    }
}

/** Decodes a vector load or store (LOAD-FP / STORE-FP with a vector
 *  width): class and sources.
 *
 *  @param inst Instruction being decoded.
 *  @param word Raw encoding.
 *  @param is_store True for a store.
 */
void decode_vector_mem(Inst& inst, std::uint32_t word, bool is_store) {
    inst.cls = is_store ? InstClass::V_STORE : InstClass::V_LOAD;
    const std::uint32_t mop = bits(word, 27, 26);
    add(inst.srcs, File::X, bits(word, 19, 15));
    if (mop == 2) { // strided
        add(inst.srcs, File::X, bits(word, 24, 20));
    }
    if (mop == 1 || mop == 3) { // indexed (coarse: one index register)
        add_v(inst.srcs, bits(word, 24, 20), 1);
    }
    if (bits(word, 25, 25) == 0) { // masked
        add_v(inst.srcs, 0, 1);
    }
    if (is_store) { // the data: nf segments of LMUL registers
        const std::uint32_t nf = bits(word, 31, 29) + 1;
        add_v(inst.srcs, bits(word, 11, 7),
              std::max<std::uint32_t>(1, inst.lmul8 / 8) * nf);
    }
}

} // namespace

void decode_inst(Inst& inst) {
    // Mnemonic = first whitespace-separated token of the disassembly.
    const std::string& text = inst.disasm;
    inst.mnemonic = text.substr(0, text.find_first_of(" \t"));

    const std::uint32_t word = inst.encoding;
    inst.srcs.clear();
    inst.cls = InstClass::UNKNOWN;

    if (const std::optional<ScalarForm> form = scalar_form(word)) {
        inst.cls = form->cls;
        add(inst.srcs, form->rs1, bits(word, 19, 15));
        add(inst.srcs, form->rs2, bits(word, 24, 20));
        add(inst.srcs, form->rs3, bits(word, 31, 27));
        return;
    }
    switch (bits(word, 6, 0)) {
    case kOpLoadFp:
        decode_vector_mem(inst, word, /*is_store=*/false);
        break;
    case kOpStoreFp:
        decode_vector_mem(inst, word, /*is_store=*/true);
        break;
    case kOpVector:
        decode_op_v(inst, word);
        break;
    default:
        break;
    }
}

} // namespace reef_perf
