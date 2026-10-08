#include "reef_perf/common/inst.hpp"

#include <ios>
#include <ostream>

namespace reef_perf {

const char* class_name(InstClass cls) {
    switch (cls) {
    case InstClass::ALU:
        return "alu";
    case InstClass::BRANCH:
        return "branch";
    case InstClass::JUMP:
        return "jump";
    case InstClass::MUL:
        return "mul";
    case InstClass::DIV:
        return "div";
    case InstClass::CSR:
        return "csr";
    case InstClass::FENCE:
        return "fence";
    case InstClass::SYSTEM:
        return "system";
    case InstClass::LOAD:
        return "load";
    case InstClass::STORE:
        return "store";
    case InstClass::FP:
        return "fp";
    case InstClass::FP_DIV:
        return "fp_div";
    case InstClass::FP_LOAD:
        return "fp_load";
    case InstClass::FP_STORE:
        return "fp_store";
    case InstClass::VSET:
        return "vset";
    case InstClass::V_ALU:
        return "v_alu";
    case InstClass::V_MUL:
        return "v_mul";
    case InstClass::V_DIV:
        return "v_div";
    case InstClass::V_FP:
        return "v_fp";
    case InstClass::V_FDIV:
        return "v_fdiv";
    case InstClass::V_PERM:
        return "v_perm";
    case InstClass::V_LOAD:
        return "v_load";
    case InstClass::V_STORE:
        return "v_store";
    case InstClass::V_TO_SCALAR:
        return "v_to_scalar";
    case InstClass::UNKNOWN:
    case InstClass::NUM_CLASSES:
        break;
    }
    return "unknown";
}

std::ostream& operator<<(std::ostream& os, const Inst& inst) {
    return os << "#" << inst.seq << " 0x" << std::hex << inst.pc << std::dec
              << " " << inst.disasm;
}

std::ostream& operator<<(std::ostream& os, const FetchPacket& pkt) {
    return os << "[" << pkt.insts.size() << " insts"
              << (pkt.last ? ", last" : "") << "]";
}

} // namespace reef_perf
