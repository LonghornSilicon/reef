//===- Dma.cpp - The DMA initiator ----------------------------------------===//

#include "memory/Dma.h"

#include "memory/MemTiming.h"
#include "memory/Tcm.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace reef_perf;

Dma::Dma(uint32_t SetupCycles, uint32_t BeatBytes, ExtMemModel *Ext,
         TcmLookup FindTcm)
    : SetupCycles(SetupCycles), BeatBytes(BeatBytes), Ext(Ext),
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
  const Cycle Begin = std::max(Arrive, getFinish()) + SetupCycles;
  const uint32_t Beats = (Len + BeatBytes - 1) / BeatBytes;
  ++Stats.Transfers;
  Stats.Bytes += Len;
  const size_t J = Jobs.size();
  Jobs.push_back({Arrive, Begin, Begin, Beats, {}, {}, {}, 0});

  // Book each end that is in a TCM: reads at SRC, writes at DST. An end in
  // external memory books the link, or costs only the setup delay when the
  // link is not modeled.
  // When each beat's bytes have come in over the link, for an external SRC.
  std::vector<Cycle> Ready;
  for (const auto &[Base, IsStore] :
       {std::pair{Src, false}, std::pair{Dst, true}}) {
    Tcm *Target = FindTcm(Base);
    if (!Target) {
      if (Ext && Ext->isTimed()) {
        const ExtMemModel::Slot Link =
            Ext->transfer(Len, std::max(Jobs[J].ExtFinish, Begin));
        Jobs[J].ExtFinish = Link.Done;
        for (uint32_t I = 0; !IsStore && I < Beats; ++I)
          Ready.push_back(Ext->getReady(
              Link, std::min<uint64_t>(Len, (I + 1) * uint64_t{BeatBytes})));
      }
      continue;
    }
    std::vector<MemAccess> Beat;
    Beat.reserve(Beats);
    for (uint32_t I = 0; I < Beats; ++I) {
      const uint32_t Size = std::min(BeatBytes, Len - (I * BeatBytes));
      Beat.push_back(
          {Base + (I * BeatBytes), static_cast<uint8_t>(Size), IsStore});
    }
    MemRequest Request;
    Request.Requester = RequesterKind::AxiSlave;
    Request.Accesses = Beat;
    Request.Earliest = Begin;
    // A later core access can push this end's last beat out; the transfer
    // and the ones queued behind it then finish later.
    const size_t End = Jobs[J].Targets.size();
    Request.OnFinishMoved = [this, J, End](Cycle Moved) {
      Jobs[J].EndFinishes[End] = Moved;
      retime(J);
    };
    // A DST beat waits for its bytes; a SRC read does not wait for the link.
    const Tcm::Booking Booked =
        Target->book(Request, IsStore ? std::span<const Cycle>(Ready)
                                      : std::span<const Cycle>());
    Jobs[J].Targets.push_back(Target);
    Jobs[J].Ids.push_back(Booked.Id);
    Jobs[J].EndFinishes.push_back(Booked.Response.Start +
                                  Booked.Response.Latency);
  }
  retime(J);
}

void Dma::retime(size_t J) {
  Job &Current = Jobs[J];
  // External to external without a link model: one beat per cycle.
  Cycle Finish = Current.EndFinishes.empty() && Current.ExtFinish == 0
                     ? Current.Begin + Current.Beats
                     : Current.ExtFinish;
  for (Cycle End : Current.EndFinishes)
    Finish = std::max(Finish, End);
  Stats.BusyCycles += Finish - Current.Finish;
  Current.Finish = Finish;
  if (J + 1 == Jobs.size())
    return;
  Job &Next = Jobs[J + 1];
  const Cycle Begin = std::max(Next.Arrive, Finish) + SetupCycles;
  if (Begin <= Next.Begin)
    return;
  Stats.BusyCycles -= Begin - Next.Begin;
  // The link stays booked where it was; only the finish moves with the start
  // (deviations A7).
  if (Next.ExtFinish != 0)
    Next.ExtFinish += Begin - Next.Begin;
  Next.Begin = Begin;
  for (size_t I = 0; I < Next.Targets.size(); ++I)
    Next.Targets[I]->delay(Next.Ids[I], Begin);
  retime(J + 1);
}
