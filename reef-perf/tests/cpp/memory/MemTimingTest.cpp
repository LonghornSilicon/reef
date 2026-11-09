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

TEST(MemoryRegionTest, ContainsIsHalfOpen) {
  MemoryRegion DTCM{
      .Start = 0x10000, .Length = 0x8000, .Kind = RegionKind::DTCM};
  EXPECT_FALSE(DTCM.contains(0xffff));
  EXPECT_TRUE(DTCM.contains(0x10000));
  EXPECT_TRUE(DTCM.contains(0x17fff));
  EXPECT_FALSE(DTCM.contains(0x18000));
}

} // namespace
