//===- Dma.cpp - The DMA initiator ----------------------------------------===//

#include "memory/Dma.h"

#include "memory/Tcm.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace reef_perf;

Dma::Dma(uint32_t SetupCycles, uint32_t BeatBytes, TcmLookup FindTcm)
    : SetupCycles(SetupCycles), BeatBytes(BeatBytes),
      FindTcm(std::move(FindTcm)) {
  if (BeatBytes == 0 || BeatBytes > 255)
    throw std::invalid_argument("dma_beat_bytes must be 1-255");
}

void Dma::writeRegister(uint32_t Offset, uint64_t Value, Cycle Arrive) {
  switch (Offset) {
  case DmaSrc:
    Src = static_cast<Address>(Value);
    break;
  case DmaDst:
    Dst = static_cast<Address>(Value);
    break;
  case DmaLen:
    Len = static_cast<uint32_t>(Value);
    break;
  case DmaGo:
    start(Arrive);
    break;
  default:
    break; // WAIT and unused offsets ignore stores.
  }
}

void Dma::start(Cycle Arrive) {
  if (Len == 0)
    return;
  const Cycle Begin = std::max(Arrive, Finish) + SetupCycles;
  const uint32_t Beats = (Len + BeatBytes - 1) / BeatBytes;
  ++Stats.Transfers;
  Stats.Bytes += Len;
  Cycle End = Begin;

  // Book each end that is in a TCM: reads at SRC, writes at DST. An end in
  // external memory costs only the setup delay.
  for (const auto &[Base, IsStore] :
       {std::pair{Src, false}, std::pair{Dst, true}}) {
    Tcm *Target = FindTcm(Base);
    if (!Target)
      continue;
    std::vector<MemAccess> Beat;
    Beat.reserve(Beats);
    for (uint32_t I = 0; I < Beats; ++I)
      Beat.push_back(
          {Base + (I * BeatBytes), static_cast<uint8_t>(BeatBytes), IsStore});
    MemRequest Request;
    Request.Requester = RequesterKind::AxiSlave;
    Request.Accesses = Beat;
    Request.Earliest = Begin;
    // A later core access can push the transfer's last beat out; the
    // transfer then finishes later.
    Request.OnFinishMoved = [this](Cycle Moved) {
      Finish = std::max(Finish, Moved);
    };
    MemResponse Response = Target->access(Request);
    End = std::max(End, Response.Start + Response.Latency);
  }
  if (End == Begin)
    End = Begin + Beats; // External to external: one beat per cycle.
  Stats.BusyCycles += End - Begin;
  Finish = End;
}
