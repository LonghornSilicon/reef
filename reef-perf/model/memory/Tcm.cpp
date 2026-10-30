//===- Tcm.cpp - A tightly coupled memory ---------------------------------===//

#include "memory/Tcm.h"

#include "memory/MemTiming.h"

#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

using namespace reef_perf;

uint32_t reef_perf::getTcmPriority(RequesterKind Kind) {
  switch (Kind) {
  case RequesterKind::IFetch:
  case RequesterKind::LSU:
  case RequesterKind::Matrix: // Not in M3; treated as part of the core.
    return 0;
  case RequesterKind::AxiSlave:
    return 1;
  case RequesterKind::Debug:
    return 2;
  }
  throw std::invalid_argument("unknown requester");
}

Tcm::Tcm(std::string Name, uint32_t Latency, uint32_t LineBytes,
         uint32_t CyclesPerLine)
    : Latency(Latency), LineBytes(LineBytes), CyclesPerLine(CyclesPerLine),
      Port(std::move(Name), 1) {}

MemResponse Tcm::access(const MemRequest &Request) {
  if (getTcmPriority(Request.Requester) != 0)
    return book(Request).Response;
  const uint64_t Occupancy =
      static_cast<uint64_t>(countDistinctBlocks(Request.Accesses, LineBytes)) *
      CyclesPerLine;
  TcmRequesterStats &S = statsFor(Request.Requester);
  const bool IsWrite =
      !Request.Accesses.empty() && Request.Accesses.front().IsStore;
  ++(IsWrite ? S.Writes : S.Reads);
  for (const MemAccess &Access : Request.Accesses)
    S.Bytes += Access.Size;
  S.PortCycles += Occupancy;
  return accessCore(Request, Occupancy);
}

Tcm::Booking Tcm::book(const MemRequest &Request) {
  if (getTcmPriority(Request.Requester) == 0)
    throw std::invalid_argument("Tcm::book takes lower-priority requests");
  // Each access is its own port transaction, so beats narrower than a line
  // still take a port cycle each.
  uint64_t Occupancy = 0;
  TcmRequesterStats &S = statsFor(Request.Requester);
  for (const MemAccess &Access : Request.Accesses) {
    Occupancy += static_cast<uint64_t>(countDistinctBlocks(
                     std::span<const MemAccess>(&Access, 1), LineBytes)) *
                 CyclesPerLine;
    S.Bytes += Access.Size;
  }
  const bool IsWrite =
      !Request.Accesses.empty() && Request.Accesses.front().IsStore;
  ++(IsWrite ? S.Writes : S.Reads);
  S.PortCycles += Occupancy;
  return accessLower(Request, Occupancy);
}

MemResponse Tcm::accessCore(const MemRequest &Request, uint64_t Occupancy) {
  MemResponse Response;
  Response.Occupancy = Occupancy;
  // The first line is usable after Latency; each further line adds
  // CyclesPerLine.
  Response.Latency = Latency + Occupancy - CyclesPerLine;
  Response.Start = Port.reserve(Request.Earliest, Occupancy);
  statsFor(Request.Requester).WaitCycles += Response.Start - Request.Earliest;
  CoreBusy[Response.Start] = Response.Start + Occupancy;
  for (Cycle C = Response.Start; C < Response.Start + Occupancy; ++C)
    if (Owner.contains(C))
      bump(C);
  return Response;
}

Tcm::Booking Tcm::accessLower(const MemRequest &Request, uint64_t Occupancy) {
  const uint32_t Priority = getTcmPriority(Request.Requester);
  const auto Id = static_cast<uint32_t>(Transfers.size());
  Transfers.push_back({Request.Requester, {}, Request.OnFinishMoved});
  for (Cycle C = Request.Earliest; Transfers[Id].Cycles.size() < Occupancy; ++C)
    if (isFreeFor(C, Priority))
      take(C, Id);

  const Transfer &T = Transfers[Id];
  MemResponse Response;
  Response.Start = *T.Cycles.begin();
  Response.Occupancy = Occupancy;
  Response.Latency = getFinish(T) - Response.Start;
  statsFor(Request.Requester).WaitCycles +=
      *T.Cycles.rbegin() + 1 - Request.Earliest - Occupancy;
  return {Response, Id};
}

void Tcm::delay(uint32_t Id, Cycle Earliest) {
  Transfer &T = Transfers[Id];
  if (T.Cycles.empty() || *T.Cycles.begin() >= Earliest)
    return;
  const Cycle OldFinish = getFinish(T);
  uint64_t Moved = 0;
  while (!T.Cycles.empty() && *T.Cycles.begin() < Earliest) {
    Owner.erase(*T.Cycles.begin());
    T.Cycles.erase(T.Cycles.begin());
    ++Moved;
  }
  const uint32_t Priority = getTcmPriority(T.Requester);
  for (Cycle C = Earliest; Moved > 0; ++C) {
    if (isFreeFor(C, Priority)) {
      take(C, Id);
      --Moved;
    }
  }
  reportMove(T, OldFinish);
}

bool Tcm::isCoreBusy(Cycle C) const {
  auto It = CoreBusy.upper_bound(C);
  return It != CoreBusy.begin() && std::prev(It)->second > C;
}

bool Tcm::isFreeFor(Cycle C, uint32_t Priority) const {
  if (isCoreBusy(C))
    return false;
  auto It = Owner.find(C);
  return It == Owner.end() ||
         getTcmPriority(Transfers[It->second].Requester) > Priority;
}

void Tcm::take(Cycle C, uint32_t Id) {
  if (Owner.contains(C))
    bump(C);
  Owner[C] = Id;
  Transfers[Id].Cycles.insert(C);
}

void Tcm::bump(Cycle C) {
  const uint32_t Id = Owner.at(C);
  Transfer &T = Transfers[Id];
  const Cycle OldFinish = getFinish(T);
  Owner.erase(C);
  T.Cycles.erase(C);
  TcmRequesterStats &S = statsFor(T.Requester);
  ++S.BumpedCycles;
  const uint32_t Priority = getTcmPriority(T.Requester);
  Cycle Next = C + 1;
  while (!isFreeFor(Next, Priority))
    ++Next;
  take(Next, Id);
  reportMove(T, OldFinish);
}

void Tcm::reportMove(Transfer &T, Cycle OldFinish) {
  const Cycle NewFinish = getFinish(T);
  statsFor(T.Requester).WaitCycles += NewFinish - OldFinish;
  if (NewFinish != OldFinish && T.OnFinishMoved)
    T.OnFinishMoved(NewFinish);
}

Cycle Tcm::getFinish(const Transfer &T) const {
  // As for the core: the last line is usable Latency cycles after it starts.
  return *T.Cycles.rbegin() + 1 - CyclesPerLine + Latency;
}

const TcmRequesterStats &Tcm::getStats(RequesterKind Kind) const {
  return Stats[static_cast<size_t>(Kind)];
}

TcmRequesterStats &Tcm::statsFor(RequesterKind Kind) {
  return Stats[static_cast<size_t>(Kind)];
}
