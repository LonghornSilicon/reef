//===- Vector.cpp - The vector module -------------------------------------===//

#include "vector/Vector.h"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string_view>

using namespace reef_perf;

uint32_t reef_perf::getVectorUops(const Inst &I, uint32_t VLENBits) {
  uint32_t VLENBytes = VLENBits / 8;
  uint32_t Bytes = I.VL * I.SEWBytes;
  return std::max<uint32_t>(1, (Bytes + VLENBytes - 1) / VLENBytes);
}

bool reef_perf::usesAddSubStage(const Inst &I) {
  static const std::set<std::string_view> Ops = {
      "vadd",   "vsub",   "vrsub",  "vadc",   "vsbc",   "vmadc",  "vmsbc",
      "vsadd",  "vsaddu", "vssub",  "vssubu", "vaadd",  "vaaddu", "vasub",
      "vasubu", "vwadd",  "vwaddu", "vwsub",  "vwsubu", "vmin",   "vminu",
      "vmax",   "vmaxu",  "vmseq",  "vmsne",  "vmslt",  "vmsltu", "vmsle",
      "vmsleu", "vmsgt",  "vmsgtu"};
  std::string_view Mnemonic = I.Mnemonic;
  return Ops.count(Mnemonic.substr(0, Mnemonic.find('.'))) != 0;
}

static VectorPipelineConfig
getPipelineConfig(const Vector::VectorParameterSet *Params) {
  VectorPipelineConfig Config;
  Config.De1Width = Params->vec_de1_width;
  Config.LegalQueueEntries = Params->vec_legal_queue_entries;
  Config.De2Width = Params->vec_de2_width;
  Config.De2UopWidth = Params->vec_de2_uop_width;
  Config.UopQueueEntries = Params->vec_uop_queue_entries;
  Config.DispatchWidth = Params->vec_dispatch_width;
  Config.StationEntries = Params->vec_station_entries;
  Config.RobEntries = Params->vec_rob_entries;
  Config.RetireWidth = Params->vec_retire_width;
  return Config;
}

Vector::Vector(sparta::TreeNode *Node, const VectorParameterSet *Params)
    : sparta::Unit(Node, name), QueueEntries(Params->vec_queue_entries),
      VLENBits(Params->vlen_bits), AddSubLatency(Params->vec_addsub_latency),
      AluLatency(Params->vec_alu_latency), MulLatency(Params->vec_mul_latency),
      PermLatency(Params->vec_perm_latency), FpLatency(Params->vec_fp_latency),
      CyclesPerUop(Params->vec_cycles_per_uop),
      DivCyclesPerUop(Params->vec_div_cycles_per_uop),
      ToScalarLatency(Params->vec_to_scalar_latency),
      VsetLatency(Params->vset_latency), Lanes("vector", Params->vec_lanes),
      Pipeline(getPipelineConfig(Params)) {
  sparta_assert(VLENBits >= 8);
  InInstsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Vector, receiveInst, InstRef));
  sparta::StartupEvent(Node, CREATE_SPARTA_HANDLER(Vector, sendInitialCredits));
}

void Vector::sendInitialCredits() { OutCreditsPort.send(QueueEntries); }

uint64_t Vector::getLatency(const Inst &I) const {
  switch (I.Class) {
  case InstClass::V_MUL:
    return MulLatency;
  case InstClass::V_PERM:
    return PermLatency;
  case InstClass::V_FP:
    return FpLatency;
  case InstClass::V_DIV:
  case InstClass::V_FDIV:
    return DivCyclesPerUop;
  default:
    return usesAddSubStage(I) ? AddSubLatency : AluLatency;
  }
}

void Vector::receiveInst(const InstRef &I) {
  Cycle Now = getClock()->currentCycle();
  ++NumExecuted;

  // vset* is resolved in the frontend and never reaches the command queue;
  // its credit returns when it leaves the frontend.
  if (I->Class == InstClass::VSET) {
    I->IssueCycle = Now;
    I->ResultReadyCycle = Now + VsetLatency - 1;
    I->CompleteCycle = Now + VsetLatency;
    OutCreditsPort.send(1);
    return;
  }

  uint32_t Uops = getVectorUops(*I, VLENBits);
  bool Divide = I->Class == InstClass::V_DIV || I->Class == InstClass::V_FDIV;
  VectorTiming T = Pipeline.schedule(I->DispatchCycle, Uops, getLatency(*I),
                                     Divide ? DivCyclesPerUop : CyclesPerUop,
                                     I->VecSourcesReadyCycle, Lanes);
  NumUops += Uops;

  I->IssueCycle = T.FirstStart;
  I->ResultReadyCycle = T.Retire;
  if (I->Class == InstClass::V_TO_SCALAR)
    I->ResultReadyCycle += ToScalarLatency;
  I->CompleteCycle = T.Retire;

  // Dispatch may use the command-queue entry the cycle after DE1 takes it.
  sparta_assert(T.CommandQueuePop >= Now);
  OutCreditsPort.send(1, T.CommandQueuePop - Now);
}
