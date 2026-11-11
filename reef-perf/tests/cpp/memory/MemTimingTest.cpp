//===- MemTimingTest.cpp - Memory timing model tests ----------------------===//

#include "memory/MemTiming.h"
#include "common/MemoryMap.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace reef_perf;

/// \returns one \p Size-byte load at each of \p Addrs.
static std::vector<MemAccess> getAccesses(const std::vector<uint32_t> &Addrs,
                                          uint8_t Size) {
  std::vector<MemAccess> Accesses;
  Accesses.reserve(Addrs.size());
  for (uint32_t Addr : Addrs)
    Accesses.push_back({Addr, Size, false});
  return Accesses;
}

namespace {

TEST(DistinctBlocksTest, CountsEachBlockOnce) {
  EXPECT_EQ(countDistinctBlocks(getAccesses({0x0, 0x4, 0x8, 0xc}, 4), 16), 1U);
  EXPECT_EQ(countDistinctBlocks(getAccesses({0x0, 0x10, 0x20}, 4), 16), 3U);
}

TEST(DistinctBlocksTest, StraddlingAccessTouchesTwoBlocks) {
  EXPECT_EQ(countDistinctBlocks(getAccesses({0xe}, 4), 16), 2U);
}

TEST(DistinctBlocksTest, EmptyListIsOneBlock) {
  EXPECT_EQ(countDistinctBlocks({}, 16), 1U);
}

TEST(DistinctBlocksTest, ZeroBlockSizeThrows) {
  EXPECT_THROW(countDistinctBlocks(getAccesses({0x0}, 4), 0),
               std::invalid_argument);
}

TEST(AxiModelTest, BeatsFollowTheBusWidth) {
  AxiModel AXI(20, 32, 4);
  MemResponse Response = AXI.access(getAccesses({0x0, 0x20}, 4), 0);
  EXPECT_EQ(Response.Start, 0U);
  EXPECT_EQ(Response.Occupancy, 2U);
  EXPECT_EQ(Response.Latency, 21U);
}

TEST(AxiModelTest, OutstandingLimitHoldsBackTheNextOperation) {
  AxiModel AXI(20, 32, 2);
  EXPECT_EQ(AXI.access(getAccesses({0x0}, 4), 0).Start, 0U);
  EXPECT_EQ(AXI.access(getAccesses({0x0}, 4), 1).Start, 1U);
  // Both slots are held until their operations finish at cycles 20 and 21.
  EXPECT_EQ(AXI.access(getAccesses({0x0}, 4), 2).Start, 20U);
}

TEST(AxiModelTest, BlockingReadHoldsItsRequesterUntilTheResponse) {
  AxiModel AXI(20, 32, 1, true);
  MemResponse Read = AXI.access(getAccesses({0x20000000}, 4), 0);
  EXPECT_EQ(Read.Occupancy, 20U); // Until the read response returns.
  EXPECT_EQ(Read.Latency, 20U);
}

TEST(AxiModelTest, BlockingReadsDoNotHoldAWrite) {
  AxiModel AXI(20, 32, 4, true);
  std::vector<MemAccess> Store = {{0x20000000, 4, true, 0}};
  EXPECT_EQ(AXI.access(Store, 0).Occupancy, 1U); // Only its one beat.
}

TEST(AxiModelTest, ExtendKeepsTheOnlySlotBusy) {
  // One slot: a store holds it for 2 cycles, then 30 more for the link.
  AxiModel AXI(2, 32, 1, true);
  std::vector<MemAccess> Store = {{0x20000000, 4, true, 0}};
  EXPECT_EQ(AXI.access(Store, 0).Start, 0U);
  AXI.extend(2, 30);
  EXPECT_EQ(AXI.access(Store, 0).Start, 32U);
}

TEST(ExtMemModelTest, WithoutARateTheLinkAddsNothing) {
  ExtMemModel Ext(26, 0);
  EXPECT_FALSE(Ext.isTimed());
  EXPECT_EQ(Ext.transfer(64, 10).Done, 10U);
}

TEST(ExtMemModelTest, FirstByteLatencyThenBytesAtTheRate) {
  // 26 cycles to the first byte, then 64 bytes at 1 B/cycle: 10 + 90.
  ExtMemModel Ext(26, 1.0);
  const ExtMemModel::Slot Booked = Ext.transfer(64, 10);
  EXPECT_EQ(Booked.FirstByte, 36U);
  EXPECT_EQ(Booked.Done, 100U);
}

TEST(ExtMemModelTest, BytesArriveAtTheRateAfterTheFirst) {
  ExtMemModel Ext(26, 1.0);
  const ExtMemModel::Slot Booked = Ext.transfer(64, 10);
  EXPECT_EQ(Ext.getReady(Booked, 16), 52U);
  EXPECT_EQ(Ext.getReady(Booked, 64), 100U);
}

TEST(ExtMemModelTest, OperationsShareOneLink) {
  ExtMemModel Ext(26, 1.0);
  EXPECT_EQ(Ext.transfer(64, 10).Done, 100U);
  EXPECT_EQ(Ext.transfer(64, 20).Done, 190U); // Waits for the first.
}

TEST(ExtMemModelTest, FractionalRateRoundsUp) {
  // 16 bytes at 2.64 B/cycle (132 MB/s at 50 MHz) is 6.06, so 7 cycles.
  ExtMemModel Ext(0, 2.64);
  EXPECT_EQ(Ext.transfer(16, 0).Done, 7U);
}

TEST(ExtMemModelTest, AWriteCanHaveItsOwnLatency) {
  ExtMemModel Ram("ram", 30, 16, 1.0, 0);
  EXPECT_EQ(Ram.transfer(8, 0).Done, 38U);          // Read: 30 + 8.
  EXPECT_EQ(Ram.transfer(8, 100, true).Done, 124U); // Write: 16 + 8.
}

TEST(ExtMemModelTest, EachBurstPaysTheLatency) {
  // 250 bytes in bursts of 100: 3 bursts x 30 cycles, and 250 data cycles.
  ExtMemModel Ram("ram", 30, 16, 1.0, 100);
  const ExtMemModel::Slot Booked = Ram.transfer(250, 0);
  EXPECT_EQ(Booked.FirstByte, 30U);
  EXPECT_EQ(Booked.Done, 340U);
  EXPECT_EQ(Ram.getReady(Booked, 100), 130U); // The first burst.
  EXPECT_EQ(Ram.getReady(Booked, 101), 161U); // After the second command.
  EXPECT_EQ(Ram.getReady(Booked, 250), 340U);
}

TEST(ExtMemModelTest, NegativeRateThrows) {
  EXPECT_THROW(ExtMemModel(0, -1.0), std::invalid_argument);
}

TEST(MemoryRegionTest, ContainsIsHalfOpen) {
  MemoryRegion DTCM{
      .Start = 0x10000, .Length = 0x8000, .Kind = RegionKind::DTCM};
  EXPECT_FALSE(DTCM.contains(0xffff));
  EXPECT_TRUE(DTCM.contains(0x10000));
  EXPECT_TRUE(DTCM.contains(0x17fff));
  EXPECT_FALSE(DTCM.contains(0x18000));
}

} // namespace
