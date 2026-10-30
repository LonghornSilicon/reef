//===- TcmTest.cpp - TCM port and arbiter tests ---------------------------===//

#include "memory/Tcm.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace reef_perf;

/// \returns one \p Size-byte access at each of \p Addrs.
static std::vector<MemAccess> getAccesses(const std::vector<uint32_t> &Addrs,
                                          uint8_t Size, bool IsStore = false) {
  std::vector<MemAccess> Accesses;
  Accesses.reserve(Addrs.size());
  for (uint32_t Addr : Addrs)
    Accesses.push_back({Addr, Size, IsStore});
  return Accesses;
}

/// \returns a request from \p Kind for \p Accesses, starting at \p Earliest.
static MemRequest getRequest(RequesterKind Kind,
                             const std::vector<MemAccess> &Accesses,
                             Cycle Earliest) {
  MemRequest Request;
  Request.Requester = Kind;
  Request.Accesses = Accesses;
  Request.Earliest = Earliest;
  return Request;
}

/// \returns \p Lines consecutive 16-byte line writes from 0x10000.
static std::vector<MemAccess> getLineWrites(uint32_t Lines) {
  std::vector<uint32_t> Addrs;
  for (uint32_t I = 0; I < Lines; ++I)
    Addrs.push_back(0x10000 + (I * 16));
  return getAccesses(Addrs, 16, true);
}

namespace {

TEST(TcmTest, PriorityIsCoreThenAxiSlaveThenDebug) {
  EXPECT_EQ(getTcmPriority(RequesterKind::IFetch), 0U);
  EXPECT_EQ(getTcmPriority(RequesterKind::LSU), 0U);
  EXPECT_EQ(getTcmPriority(RequesterKind::AxiSlave), 1U);
  EXPECT_EQ(getTcmPriority(RequesterKind::Debug), 2U);
}

TEST(TcmTest, SingleLineAccessHasBaseLatency) {
  Tcm DTCM("dtcm", 2, 16, 1);
  auto Accesses = getAccesses({0x10000}, 4);
  MemResponse Response =
      DTCM.access(getRequest(RequesterKind::LSU, Accesses, 5));
  EXPECT_EQ(Response.Start, 5U);
  EXPECT_EQ(Response.Occupancy, 1U);
  EXPECT_EQ(Response.Latency, 2U);
}

TEST(TcmTest, EachExtraLineAddsOccupancyAndLatency) {
  Tcm DTCM("dtcm", 2, 16, 1);
  auto Accesses = getAccesses({0x10000, 0x10010, 0x10020, 0x10030}, 4);
  MemResponse Response =
      DTCM.access(getRequest(RequesterKind::LSU, Accesses, 0));
  EXPECT_EQ(Response.Occupancy, 4U);
  EXPECT_EQ(Response.Latency, 5U);
}

TEST(TcmTest, BusyPortDelaysTheNextCoreAccess) {
  Tcm DTCM("dtcm", 2, 16, 3);
  auto Accesses = getAccesses({0x10000}, 4);
  EXPECT_EQ(DTCM.access(getRequest(RequesterKind::LSU, Accesses, 0)).Start, 0U);
  EXPECT_EQ(DTCM.access(getRequest(RequesterKind::LSU, Accesses, 1)).Start, 3U);
  EXPECT_EQ(DTCM.getStats(RequesterKind::LSU).WaitCycles, 2U);
}

TEST(TcmTest, AxiSlaveUsesTheCyclesTheCoreLeavesFree) {
  Tcm DTCM("dtcm", 1, 16, 1);
  auto Load = getAccesses({0x10000}, 4);
  DTCM.access(getRequest(RequesterKind::LSU, Load, 0));
  DTCM.access(getRequest(RequesterKind::LSU, Load, 2));
  // Four line writes from cycle 0: the core holds 0 and 2, so the DMA gets
  // 1, 3, 4 and 5.
  auto Writes = getLineWrites(4);
  MemResponse Response =
      DTCM.access(getRequest(RequesterKind::AxiSlave, Writes, 0));
  EXPECT_EQ(Response.Start, 1U);
  EXPECT_EQ(Response.Occupancy, 4U);
  EXPECT_EQ(Response.Start + Response.Latency, 6U); // Last cycle 5, latency 1.
  EXPECT_EQ(DTCM.getStats(RequesterKind::AxiSlave).WaitCycles, 2U);
}

TEST(TcmTest, CoreBumpsAnEarlierAxiSlaveBookingAndTellsItsRequester) {
  Tcm DTCM("dtcm", 1, 16, 1);
  Cycle MovedTo = 0;
  auto Writes = getLineWrites(4);
  MemRequest Dma = getRequest(RequesterKind::AxiSlave, Writes, 100);
  Dma.OnFinishMoved = [&MovedTo](Cycle Finish) { MovedTo = Finish; };
  MemResponse Booked = DTCM.access(Dma);
  EXPECT_EQ(Booked.Start + Booked.Latency, 104U); // Cycles 100-103.

  // The core claims cycle 101 later in simulation order and still gets it.
  auto Load = getAccesses({0x10000}, 4);
  EXPECT_EQ(DTCM.access(getRequest(RequesterKind::LSU, Load, 101)).Start, 101U);
  // The DMA's line moves from 101 to 104, so it finishes a cycle later.
  EXPECT_EQ(MovedTo, 105U);
  EXPECT_EQ(DTCM.getStats(RequesterKind::AxiSlave).BumpedCycles, 1U);
}

TEST(TcmTest, AxiSlaveBumpsDebugButNotTheOtherWayRound) {
  Tcm DTCM("dtcm", 1, 16, 1);
  Cycle DebugFinish = 0;
  auto One = getLineWrites(1);
  MemRequest Debug = getRequest(RequesterKind::Debug, One, 10);
  Debug.OnFinishMoved = [&DebugFinish](Cycle Finish) { DebugFinish = Finish; };
  DTCM.access(Debug);
  MemResponse Dma = DTCM.access(getRequest(RequesterKind::AxiSlave, One, 10));
  EXPECT_EQ(Dma.Start, 10U);
  EXPECT_EQ(DebugFinish, 12U); // Moved to cycle 11.

  MemResponse Later = DTCM.access(getRequest(RequesterKind::Debug, One, 10));
  EXPECT_EQ(Later.Start, 12U); // 10 is the DMA's, 11 the first debug write's.
}

TEST(TcmTest, FetchKeepsTheHostOffItsCycles) {
  Tcm ITCM("itcm", 1, 16, 1);
  auto Line = getAccesses({0x0}, 16);
  // Fetch reads a line every other cycle: 0, 2, 4.
  for (Cycle C : {0, 2, 4})
    EXPECT_EQ(ITCM.access(getRequest(RequesterKind::IFetch, Line, C)).Start, C);
  // A 3-line host write from cycle 0 gets the gaps: 1, 3 and 5.
  auto Writes = getLineWrites(3);
  MemResponse Host =
      ITCM.access(getRequest(RequesterKind::AxiSlave, Writes, 0));
  EXPECT_EQ(Host.Start, 1U);
  EXPECT_EQ(Host.Start + Host.Latency, 6U);
}

TEST(TcmTest, FetchDoesNotDelayTheLsu) {
  Tcm ITCM("itcm", 1, 16, 1);
  auto Line = getAccesses({0x0}, 16);
  ITCM.access(getRequest(RequesterKind::IFetch, Line, 7));
  // An LSU load from the ITCM wins over fetch in the RTL.
  EXPECT_EQ(ITCM.access(getRequest(RequesterKind::LSU, Line, 7)).Start, 7U);
}

TEST(TcmTest, CountsReadsWritesAndBytesPerRequester) {
  Tcm DTCM("dtcm", 1, 16, 1);
  auto Load = getAccesses({0x10000, 0x10004}, 4);
  auto Store = getAccesses({0x10010}, 4, true);
  DTCM.access(getRequest(RequesterKind::LSU, Load, 0));
  DTCM.access(getRequest(RequesterKind::LSU, Store, 1));
  const TcmRequesterStats &S = DTCM.getStats(RequesterKind::LSU);
  EXPECT_EQ(S.Reads, 1U);
  EXPECT_EQ(S.Writes, 1U);
  EXPECT_EQ(S.Bytes, 12U);
  EXPECT_EQ(S.PortCycles, 2U);
}

} // namespace
