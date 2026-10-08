#include "reef_perf/resource_pool.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace reef_perf {

TEST(ResourcePoolTest, PipelinedUnitAcceptsOneOperationPerCycle) {
    ResourcePool pool("alu", 1);
    EXPECT_EQ(pool.reserve(10, 1), 10U);
    EXPECT_EQ(pool.reserve(10, 1), 11U);
    EXPECT_EQ(pool.reserve(10, 1), 12U);
}

TEST(ResourcePoolTest, SeveralUnitsStartTogether) {
    ResourcePool pool("alu", 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(pool.reserve(5, 1), 5U);
    }
    EXPECT_EQ(pool.reserve(5, 1), 6U);
}

TEST(ResourcePoolTest, BlockingUnitDelaysTheNextOperation) {
    ResourcePool pool("div", 1);
    EXPECT_EQ(pool.reserve(0, 32), 0U);
    EXPECT_EQ(pool.reserve(1, 32), 32U);
}

TEST(ResourcePoolTest, IdleUnitStartsAtTheRequestedCycle) {
    ResourcePool pool("lsu", 1);
    EXPECT_EQ(pool.reserve(0, 2), 0U);
    EXPECT_EQ(pool.reserve(100, 2), 100U);
}

TEST(ResourcePoolTest, CountsOperationsAndBusyCycles) {
    ResourcePool pool("mul", 2);
    pool.reserve(0, 3);
    pool.reserve(0, 3);
    pool.reserve(0, 1);
    EXPECT_EQ(pool.name(), "mul");
    EXPECT_EQ(pool.count(), 2U);
    EXPECT_EQ(pool.ops(), 3U);
    EXPECT_EQ(pool.busy_cycles(), 7U);
}

TEST(ResourcePoolTest, RejectsEmptyPool) {
    EXPECT_THROW(ResourcePool("none", 0), std::invalid_argument);
}

} // namespace reef_perf
