//===- ScalarExec.cpp - Scalar functional units ---------------------------===//

#include "core/ScalarExec.h"

#include <bit>
#include <cstdint>

using namespace reef_perf;

namespace {

uint32_t getM3DivIterations(const Inst &I) {
  if (I.Rs2Val == 0)
    return 32;

  uint32_t Dividend = I.Rs1Val;
  bool IsSigned = I.Mnemonic == "div" || I.Mnemonic == "rem";
  if (IsSigned && (Dividend & 0x80000000U))
    Dividend = ~Dividend + 1U;

  if (Dividend == 0)
    return 32;

  uint32_t LeadingZeros = std::countl_zero(Dividend);
  uint32_t Clz = LeadingZeros == 0 ? 0 : LeadingZeros - 1;
  return 32 - Clz;
}

} // namespace

ScalarExec::ScalarExec(sparta::TreeNode *Node,
                       const ScalarExecParameterSet *Params)
    : sparta::Unit(Node, name), ALULatency(Params->alu_latency),
      MULLatency(Params->mul_latency), MULOccupancy(Params->mul_occupancy),
      DIVLatency(Params->div_latency), DIVOccupancy(Params->div_occupancy),
      DIVDataDependent(Params->div_data_dependent), DIVBase(Params->div_base),
      FPULatency(Params->fpu_latency), FPUOccupancy(Params->fpu_occupancy),
      FDIVLatency(Params->fdiv_latency), FDIVOccupancy(Params->fdiv_occupancy),
      ALU("alu", Params->alu_count), MUL("mul", Params->mul_count),
      DIV("div", 1), FPU("fpu", 1), FDIV("fdiv", 1) {
  InInstsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(ScalarExec, receiveInst, InstRef));
}

void ScalarExec::receiveInst(const InstRef &I) {
  Cycle Now = getClock()->currentCycle();
  ++NumExecuted;

  ResourcePool *Pool = &ALU;
  uint64_t Occupancy = 1;
  uint64_t Latency = ALULatency;
  switch (I->Class) {
  case InstClass::MUL:
    Pool = &MUL;
    Occupancy = MULOccupancy;
    Latency = MULLatency;
    break;
  case InstClass::DIV:
    Pool = &DIV;
    if (DIVDataDependent) {
      Latency = DIVBase + getM3DivIterations(*I);
      Occupancy = Latency;
    } else {
      Latency = DIVLatency;
      Occupancy = DIVOccupancy;
    }
    break;
  case InstClass::FP:
    Pool = &FPU;
    Occupancy = FPUOccupancy;
    Latency = FPULatency;
    break;
  case InstClass::FP_DIV:
    Pool = &FDIV;
    Occupancy = FDIVOccupancy;
    Latency = FDIVLatency;
    break;
  case InstClass::UNKNOWN:
    ++NumUnknown;
    break;
  default: // ALU, BRANCH, JUMP, CSR, FENCE, SYSTEM.
    break;
  }

  Cycle Start = Pool->reserve(Now, Occupancy);
  I->IssueCycle = Start;
  I->ResultReadyCycle = Start + Latency - 1;
  I->CompleteCycle = Start + Latency;
}
