//===- Inst.cpp - The instruction object ----------------------------------===//

#include "common/Inst.h"

#include <cstddef>
#include <cstdint>
#include <ios>
#include <ostream>
#include <span>
#include <stdexcept>
#include <vector>

using namespace reef_perf;

const char *reef_perf::getClassName(InstClass Class) {
  switch (Class) {
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
  case InstClass::MATRIX:
    return "matrix";
  case InstClass::UNKNOWN:
  case InstClass::NUM_CLASSES:
    break;
  }
  return "unknown";
}

std::ostream &reef_perf::operator<<(std::ostream &OS, const Inst &I) {
  return OS << "#" << I.Seq << " 0x" << std::hex << I.PC << std::dec << " "
            << I.Disasm;
}

std::vector<std::span<const MemAccess>>
reef_perf::splitByRegister(std::span<const MemAccess> Accesses,
                           uint32_t RegBytes) {
  if (RegBytes == 0)
    throw std::invalid_argument("register size must be at least 1 byte");
  std::vector<std::span<const MemAccess>> Groups;
  size_t Begin = 0;
  uint32_t Bytes = 0;
  for (size_t Idx = 0; Idx < Accesses.size(); ++Idx) {
    Bytes += Accesses[Idx].Size;
    if (Bytes >= RegBytes || Idx + 1 == Accesses.size()) {
      Groups.push_back(Accesses.subspan(Begin, Idx + 1 - Begin));
      Begin = Idx + 1;
      Bytes = 0;
    }
  }
  return Groups;
}
