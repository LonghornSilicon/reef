//===- Matrix.cpp - The matrix module (stub) ------------------------------===//

#include "matrix/Matrix.h"

#include "sparta/events/StartupEvent.hpp"

#include <cstdint>

using namespace reef_perf;

Matrix::Matrix(sparta::TreeNode *Node, const MatrixParameterSet *Params)
    : sparta::Unit(Node, name), QueueEntries(Params->mtx_queue_entries),
      Latency(Params->mtx_latency), Occupancy(Params->mtx_occupancy),
      Engines("matrix", Params->mtx_units) {
  InInstsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Matrix, receiveInst, InstRef));
  sparta::StartupEvent(Node, CREATE_SPARTA_HANDLER(Matrix, sendInitialCredits));
}

void Matrix::sendInitialCredits() { OutCreditsPort.send(QueueEntries); }

void Matrix::receiveInst(const InstRef &I) {
  Cycle Now = getClock()->currentCycle();
  ++NumExecuted;

  Cycle Start = Engines.reserve(Now, Occupancy);
  I->IssueCycle = Start;
  I->ResultReadyCycle = Start + Latency - 1;
  I->CompleteCycle = Start + Latency;

  // The instruction leaves the command queue when it starts.
  OutCreditsPort.send(1, Start - Now);
}
