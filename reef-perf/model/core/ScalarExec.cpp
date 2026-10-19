//===- ScalarExec.cpp - Scalar functional units ---------------------------===//

#include "core/ScalarExec.h"

#include <cstdint>

using namespace reef_perf;

ScalarExec::ScalarExec(sparta::TreeNode *Node,
                       const ScalarExecParameterSet *Params)
    : sparta::Unit(Node, name), ALULatency(Params->alu_latency),
      MULLatency(Params->mul_latency), MULOccupancy(Params->mul_occupancy),
      DIVLatency(Params->div_latency), DIVOccupancy(Params->div_occupancy),
      FPULatency(Params->fpu_latency), FPUOccupancy(Params->fpu_occupancy),
      FDIVLatency(Params->fdiv_latency), FDIVOccupancy(Params->fdiv_occupancy),
      ALU("alu", Params->alu_count), MUL("mul", Params->mul_count),
      DIV("div", 1), FPU("fpu", 1), FDIV("fdiv", 1) {
  InInstsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(ScalarExec, receiveInst, InstRef));
}

void ScalarExec::receiveInst(const InstRef &I) {
  uint64_t Now = getClock()->currentCycle();
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
    Occupancy = DIVOccupancy;
    Latency = DIVLatency;
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

  uint64_t Start = Pool->reserve(Now, Occupancy);
  I->IssueCycle = Start;
  I->ResultReadyCycle = Start + Latency - 1;
  I->CompleteCycle = Start + Latency;
}
