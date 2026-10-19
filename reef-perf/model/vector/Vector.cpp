//===- Vector.cpp - The vector module -------------------------------------===//

#include "vector/Vector.h"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <algorithm>
#include <cstdint>

using namespace reef_perf;

uint32_t reef_perf::getVectorUops(const Inst &I, uint32_t VLENBits) {
  uint32_t VLENBytes = VLENBits / 8;
  uint32_t Bytes = I.VL * I.SEWBytes;
  return std::max<uint32_t>(1, (Bytes + VLENBytes - 1) / VLENBytes);
}

Vector::Vector(sparta::TreeNode *Node, const VectorParameterSet *Params)
    : sparta::Unit(Node, name), QueueEntries(Params->vec_queue_entries),
      VLENBits(Params->vlen_bits), Latency(Params->vec_latency),
      CyclesPerUop(Params->vec_cycles_per_uop),
      DivCyclesPerUop(Params->vec_div_cycles_per_uop),
      ToScalarLatency(Params->vec_to_scalar_latency),
      VsetLatency(Params->vset_latency), Lanes("vector", Params->vec_lanes) {
  sparta_assert(VLENBits >= 8);
  InInstsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Vector, receiveInst, InstRef));
  sparta::StartupEvent(Node, CREATE_SPARTA_HANDLER(Vector, sendInitialCredits));
}

void Vector::sendInitialCredits() { OutCreditsPort.send(QueueEntries); }

void Vector::receiveInst(const InstRef &I) {
  Cycle Now = getClock()->currentCycle();
  ++NumExecuted;

  Cycle Start = Now;
  uint64_t InstLatency = VsetLatency;
  if (I->Class != InstClass::VSET) {
    bool Divide = I->Class == InstClass::V_DIV || I->Class == InstClass::V_FDIV;
    uint64_t Occupancy = static_cast<uint64_t>(getVectorUops(*I, VLENBits)) *
                         (Divide ? DivCyclesPerUop : CyclesPerUop);
    InstLatency = Latency + Occupancy - 1;
    if (I->Class == InstClass::V_TO_SCALAR)
      InstLatency += ToScalarLatency;
    Start = Lanes.reserve(Now, Occupancy);
  }
  I->IssueCycle = Start;
  I->ResultReadyCycle = Start + InstLatency - 1;
  I->CompleteCycle = Start + InstLatency;

  // The instruction leaves the command queue when it starts.
  OutCreditsPort.send(1, Start - Now);
}
