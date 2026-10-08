#include "reef_perf/common/memory_map.hpp"
#include "reef_perf/mem/mem_timing.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace reef_perf {

namespace {

/** A list of accesses of the same size.
 *
 *  @param addrs Start addresses.
 *  @param size Bytes per access.
 *  @return The accesses.
 */
std::vector<MemAccess> accesses(const std::vector<std::uint32_t>& addrs,
                                std::uint8_t size) {
    std::vector<MemAccess> out;
    for (const std::uint32_t addr : addrs) {
        out.push_back({addr, size, false});
    }
    return out;
}

} // namespace

TEST(DistinctBlocksTest, CountsEachBlockOnce) {
    EXPECT_EQ(distinct_blocks(accesses({0x0, 0x4, 0x8, 0xc}, 4), 16), 1U);
    EXPECT_EQ(distinct_blocks(accesses({0x0, 0x10, 0x20}, 4), 16), 3U);
}

TEST(DistinctBlocksTest, StraddlingAccessTouchesTwoBlocks) {
    EXPECT_EQ(distinct_blocks(accesses({0xe}, 4), 16), 2U);
}

TEST(DistinctBlocksTest, EmptyListIsOneBlock) {
    EXPECT_EQ(distinct_blocks({}, 16), 1U);
}

TEST(DistinctBlocksTest, ZeroBlockSizeThrows) {
    EXPECT_THROW(distinct_blocks(accesses({0x0}, 4), 0), std::invalid_argument);
}

TEST(TcmModelTest, SingleLineAccessHasBaseLatency) {
    TcmModel tcm("dtcm", 2, 16, 1);
    const MemResponse resp = tcm.access(accesses({0x10000}, 4), 5);
    EXPECT_EQ(resp.start, 5U);
    EXPECT_EQ(resp.occupancy, 1U);
    EXPECT_EQ(resp.latency, 2U);
}

TEST(TcmModelTest, EachExtraLineAddsOccupancyAndLatency) {
    TcmModel tcm("dtcm", 2, 16, 1);
    const MemResponse resp =
        tcm.access(accesses({0x10000, 0x10010, 0x10020, 0x10030}, 4), 0);
    EXPECT_EQ(resp.occupancy, 4U);
    EXPECT_EQ(resp.latency, 5U);
}

TEST(TcmModelTest, BusyPortDelaysTheNextAccess) {
    TcmModel tcm("dtcm", 2, 16, 3);
    EXPECT_EQ(tcm.access(accesses({0x10000}, 4), 0).start, 0U);
    EXPECT_EQ(tcm.access(accesses({0x10000}, 4), 1).start, 3U);
}

TEST(AxiModelTest, BeatsFollowTheBusWidth) {
    AxiModel axi(20, 32, 4);
    const MemResponse resp = axi.access(accesses({0x0, 0x20}, 4), 0);
    EXPECT_EQ(resp.start, 0U);
    EXPECT_EQ(resp.occupancy, 2U);
    EXPECT_EQ(resp.latency, 21U);
}

TEST(AxiModelTest, OutstandingLimitHoldsBackTheNextOperation) {
    AxiModel axi(20, 32, 2);
    EXPECT_EQ(axi.access(accesses({0x0}, 4), 0).start, 0U);
    EXPECT_EQ(axi.access(accesses({0x0}, 4), 1).start, 1U);
    // Both slots are held until their operations finish at cycles 20, 21.
    EXPECT_EQ(axi.access(accesses({0x0}, 4), 2).start, 20U);
}

TEST(MemoryRegionTest, ContainsIsHalfOpen) {
    const MemoryRegion dtcm{0x10000, 0x8000, RegionKind::DTCM};
    EXPECT_FALSE(dtcm.contains(0xffff));
    EXPECT_TRUE(dtcm.contains(0x10000));
    EXPECT_TRUE(dtcm.contains(0x17fff));
    EXPECT_FALSE(dtcm.contains(0x18000));
}

} // namespace reef_perf
