#include "InstDecode.hpp"

#include <algorithm>
#include <string_view>

namespace coralnpu_perf {

const char* className(InstClass c) {
  switch (c) {
    case InstClass::ALU: return "alu";
    case InstClass::BRANCH: return "branch";
    case InstClass::JUMP: return "jump";
    case InstClass::MUL: return "mul";
    case InstClass::DIV: return "div";
    case InstClass::CSR: return "csr";
    case InstClass::FENCE: return "fence";
    case InstClass::SYSTEM: return "system";
    case InstClass::LOAD: return "load";
    case InstClass::STORE: return "store";
    case InstClass::FP: return "fp";
    case InstClass::FP_DIV: return "fp_div";
    case InstClass::FP_LOAD: return "fp_load";
    case InstClass::FP_STORE: return "fp_store";
    case InstClass::VSET: return "vset";
    case InstClass::V_ALU: return "v_alu";
    case InstClass::V_MUL: return "v_mul";
    case InstClass::V_DIV: return "v_div";
    case InstClass::V_FP: return "v_fp";
    case InstClass::V_FDIV: return "v_fdiv";
    case InstClass::V_PERM: return "v_perm";
    case InstClass::V_LOAD: return "v_load";
    case InstClass::V_STORE: return "v_store";
    case InstClass::V_TO_SCALAR: return "v_to_scalar";
    default: return "unknown";
  }
}

namespace {

inline uint32_t bits(uint32_t w, int hi, int lo) {
  return (w >> lo) & ((1u << (hi - lo + 1)) - 1);
}

bool startsWith(std::string_view s, std::string_view prefix) {
  return s.substr(0, prefix.size()) == prefix;
}

void addX(std::vector<uint16_t>& v, uint32_t r) {
  if (r != 0) v.push_back(kXBase + r);  // x0 is never a dependency
}
void addF(std::vector<uint16_t>& v, uint32_t r) { v.push_back(kFBase + r); }
void addV(std::vector<uint16_t>& v, uint32_t base, uint32_t count) {
  for (uint32_t k = 0; k < count && base + k < 32; ++k) {
    v.push_back(kVBase + base + k);
  }
}

// Class of an OP-V instruction, from its mnemonic.
InstClass vectorArithClass(std::string_view m) {
  if (startsWith(m, "vmv.x.s") || startsWith(m, "vcpop") ||
      startsWith(m, "vfirst") || startsWith(m, "vfmv.f.s")) {
    return InstClass::V_TO_SCALAR;
  }
  if (startsWith(m, "vred") || startsWith(m, "vwred") ||
      startsWith(m, "vfred") || startsWith(m, "vfwred") ||
      startsWith(m, "vrgather") || startsWith(m, "vslide") ||
      startsWith(m, "vfslide") || startsWith(m, "vcompress")) {
    return InstClass::V_PERM;
  }
  if (startsWith(m, "vdiv") || startsWith(m, "vrem")) return InstClass::V_DIV;
  if (startsWith(m, "vfdiv") || startsWith(m, "vfrdiv") ||
      startsWith(m, "vfsqrt")) {
    return InstClass::V_FDIV;
  }
  if (startsWith(m, "vmul") || startsWith(m, "vmacc") ||
      startsWith(m, "vnmsac") || startsWith(m, "vmadd") ||
      startsWith(m, "vnmsub") || startsWith(m, "vwmul") ||
      startsWith(m, "vwmacc") || startsWith(m, "vsmul")) {
    return InstClass::V_MUL;
  }
  if (startsWith(m, "vf")) return InstClass::V_FP;
  return InstClass::V_ALU;
}

// OP-V (major opcode 0x57).
void decodeOpV(Inst& inst, uint32_t w) {
  const uint32_t funct3 = bits(w, 14, 12);
  const uint32_t funct6 = bits(w, 31, 26);
  const uint32_t vm = bits(w, 25, 25);
  const uint32_t vs2 = bits(w, 24, 20);
  const uint32_t rs1 = bits(w, 19, 15);  // also vs1
  const uint32_t rd = bits(w, 11, 7);    // also vd
  std::string_view m = inst.mnemonic;

  if (funct3 == 7) {  // vsetvli / vsetivli / vsetvl
    inst.cls = InstClass::VSET;
    if (bits(w, 31, 30) != 3) addX(inst.srcs, rs1);   // not vsetivli
    if (bits(w, 31, 25) == 0x40) addX(inst.srcs, vs2);  // vsetvl reads rs2
    addX(inst.dsts, rd);
    return;
  }

  inst.cls = vectorArithClass(m);

  // Registers per operand group. Coarse: every operand spans LMUL registers;
  // widening results span 2*LMUL. Mask-producing ops are not special-cased.
  const uint32_t emul = std::max<uint32_t>(1, inst.lmul8 / 8);
  const bool widening = startsWith(m, "vw") || startsWith(m, "vfw");
  const bool narrowing = startsWith(m, "vn") || startsWith(m, "vfn");
  const uint32_t dst_regs = widening ? 2 * emul : emul;
  const uint32_t vs2_regs = narrowing ? 2 * emul : emul;

  // Sources.
  const bool unary_wx = (funct3 == 2 && funct6 == 0x10);  // vmv.x.s, vcpop, vfirst
  const bool unary_wf = (funct3 == 1 && funct6 == 0x10);  // vfmv.f.s
  const bool scalar_to_vec = (funct3 == 6 || funct3 == 5) && funct6 == 0x10;  // vmv.s.x, vfmv.s.f
  switch (funct3) {
    case 0:  // OPIVV
    case 1:  // OPFVV
    case 2:  // OPMVV
      if (!unary_wx && !unary_wf) addV(inst.srcs, rs1, emul);
      addV(inst.srcs, vs2, vs2_regs);
      break;
    case 3:  // OPIVI
      addV(inst.srcs, vs2, vs2_regs);
      break;
    case 4:  // OPIVX
    case 6:  // OPMVX
      addX(inst.srcs, rs1);
      if (!scalar_to_vec) addV(inst.srcs, vs2, vs2_regs);
      break;
    case 5:  // OPFVF
      addF(inst.srcs, rs1);
      if (!scalar_to_vec) addV(inst.srcs, vs2, vs2_regs);
      break;
  }
  if (vm == 0) addV(inst.srcs, 0, 1);  // masked: reads v0

  // Multiply-accumulate forms also read the destination.
  const bool int_mac = (funct3 == 2 || funct3 == 6) &&
                       (funct6 == 0x29 || funct6 == 0x2b || funct6 == 0x2d ||
                        funct6 == 0x2f || funct6 >= 0x3c);
  const bool fp_mac = (funct3 == 1 || funct3 == 5) &&
                      ((funct6 >= 0x28 && funct6 <= 0x2f) || funct6 >= 0x3c);
  if (int_mac || fp_mac) addV(inst.srcs, rd, dst_regs);

  // Destination.
  if (unary_wx) {
    addX(inst.dsts, rd);
  } else if (unary_wf) {
    addF(inst.dsts, rd);
  } else {
    addV(inst.dsts, rd, dst_regs);
  }
}

// LOAD-FP / STORE-FP (0x07 / 0x27): scalar FP or vector memory ops.
void decodeFpOrVectorMem(Inst& inst, uint32_t w, bool is_store) {
  const uint32_t width = bits(w, 14, 12);
  const uint32_t rs1 = bits(w, 19, 15);
  const uint32_t rs2 = bits(w, 24, 20);  // stride reg / index vreg / fp store data
  const uint32_t rd = bits(w, 11, 7);    // load dst / vector store data
  if (width >= 1 && width <= 4) {        // flh/flw/fld/flq (M3 has flw/fsw)
    inst.cls = is_store ? InstClass::FP_STORE : InstClass::FP_LOAD;
    addX(inst.srcs, rs1);
    if (is_store) {
      addF(inst.srcs, rs2);
    } else {
      addF(inst.dsts, rd);
    }
    return;
  }
  inst.cls = is_store ? InstClass::V_STORE : InstClass::V_LOAD;
  const uint32_t mop = bits(w, 27, 26);
  const uint32_t nf = bits(w, 31, 29) + 1;
  const uint32_t emul = std::max<uint32_t>(1, inst.lmul8 / 8) * nf;
  addX(inst.srcs, rs1);
  if (mop == 2) addX(inst.srcs, rs2);                 // strided
  if (mop == 1 || mop == 3) addV(inst.srcs, rs2, 1);  // indexed (coarse)
  if (bits(w, 25, 25) == 0) addV(inst.srcs, 0, 1);    // masked
  if (is_store) {
    addV(inst.srcs, rd, emul);
  } else {
    addV(inst.dsts, rd, emul);
  }
}

// OP-FP (0x53).
void decodeOpFp(Inst& inst, uint32_t w) {
  const uint32_t funct5 = bits(w, 31, 27);
  const uint32_t rs1 = bits(w, 19, 15);
  const uint32_t rs2 = bits(w, 24, 20);
  const uint32_t rd = bits(w, 11, 7);
  inst.cls = InstClass::FP;
  switch (funct5) {
    case 0x03:  // fdiv.s
      inst.cls = InstClass::FP_DIV;
      addF(inst.srcs, rs1);
      addF(inst.srcs, rs2);
      addF(inst.dsts, rd);
      break;
    case 0x0b:  // fsqrt.s
      inst.cls = InstClass::FP_DIV;
      addF(inst.srcs, rs1);
      addF(inst.dsts, rd);
      break;
    case 0x14:  // feq/flt/fle -> x
      addF(inst.srcs, rs1);
      addF(inst.srcs, rs2);
      addX(inst.dsts, rd);
      break;
    case 0x18:  // fcvt.w[u].s -> x
    case 0x1c:  // fmv.x.w, fclass.s -> x
      addF(inst.srcs, rs1);
      addX(inst.dsts, rd);
      break;
    case 0x1a:  // fcvt.s.w[u] <- x
    case 0x1e:  // fmv.w.x <- x
      addX(inst.srcs, rs1);
      addF(inst.dsts, rd);
      break;
    case 0x08:  // fcvt.s.d etc. (unary)
      addF(inst.srcs, rs1);
      addF(inst.dsts, rd);
      break;
    default:  // fadd, fsub, fmul, fsgnj*, fmin/fmax
      addF(inst.srcs, rs1);
      addF(inst.srcs, rs2);
      addF(inst.dsts, rd);
      break;
  }
}

}  // namespace

void decodeInst(Inst& inst) {
  // Mnemonic = first whitespace-separated token of the disassembly.
  const std::string& d = inst.disasm;
  const size_t end = d.find_first_of(" \t");
  inst.mnemonic = d.substr(0, end);

  const uint32_t w = inst.encoding;
  const uint32_t opcode = bits(w, 6, 0);
  const uint32_t rd = bits(w, 11, 7);
  const uint32_t funct3 = bits(w, 14, 12);
  const uint32_t rs1 = bits(w, 19, 15);
  const uint32_t rs2 = bits(w, 24, 20);
  const uint32_t funct7 = bits(w, 31, 25);

  inst.srcs.clear();
  inst.dsts.clear();

  switch (opcode) {
    case 0x37:  // lui
    case 0x17:  // auipc
      inst.cls = InstClass::ALU;
      addX(inst.dsts, rd);
      break;
    case 0x6f:  // jal
      inst.cls = InstClass::JUMP;
      addX(inst.dsts, rd);
      break;
    case 0x67:  // jalr
      inst.cls = InstClass::JUMP;
      addX(inst.srcs, rs1);
      addX(inst.dsts, rd);
      break;
    case 0x63:  // branches
      inst.cls = InstClass::BRANCH;
      addX(inst.srcs, rs1);
      addX(inst.srcs, rs2);
      break;
    case 0x03:  // loads
      inst.cls = InstClass::LOAD;
      addX(inst.srcs, rs1);
      addX(inst.dsts, rd);
      break;
    case 0x23:  // stores
      inst.cls = InstClass::STORE;
      addX(inst.srcs, rs1);
      addX(inst.srcs, rs2);
      break;
    case 0x13:  // OP-IMM
      inst.cls = InstClass::ALU;
      addX(inst.srcs, rs1);
      addX(inst.dsts, rd);
      break;
    case 0x33:  // OP
      if (funct7 == 0x01) {
        inst.cls = funct3 < 4 ? InstClass::MUL : InstClass::DIV;
      } else {
        inst.cls = InstClass::ALU;
      }
      addX(inst.srcs, rs1);
      addX(inst.srcs, rs2);
      addX(inst.dsts, rd);
      break;
    case 0x0f:  // fence, fence.i
      inst.cls = InstClass::FENCE;
      break;
    case 0x73:  // SYSTEM
      if (funct3 == 0) {
        inst.cls = InstClass::SYSTEM;  // ecall, ebreak, mret, wfi, mpause
      } else {
        inst.cls = InstClass::CSR;
        if (funct3 <= 3) addX(inst.srcs, rs1);  // register (not immediate) forms
        addX(inst.dsts, rd);
      }
      break;
    case 0x07:  // LOAD-FP
      decodeFpOrVectorMem(inst, w, /*is_store=*/false);
      break;
    case 0x27:  // STORE-FP
      decodeFpOrVectorMem(inst, w, /*is_store=*/true);
      break;
    case 0x43:  // fmadd.s
    case 0x47:  // fmsub.s
    case 0x4b:  // fnmsub.s
    case 0x4f:  // fnmadd.s
      inst.cls = InstClass::FP;
      addF(inst.srcs, rs1);
      addF(inst.srcs, rs2);
      addF(inst.srcs, bits(w, 31, 27));
      addF(inst.dsts, rd);
      break;
    case 0x53:
      decodeOpFp(inst, w);
      break;
    case 0x57:
      decodeOpV(inst, w);
      break;
    default:
      inst.cls = InstClass::UNKNOWN;
      break;
  }
}

}  // namespace coralnpu_perf
