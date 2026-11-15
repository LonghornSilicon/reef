//===- ResourcePoolTest.cpp - ResourcePool tests --------------------------===//

#include "common/ResourcePool.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace reef_perf;

namespace {

TEST(ResourcePoolTest, PipelinedUnitAcceptsOneOperationPerCycle) {
  ResourcePool Pool("alu", 1);
  EXPECT_EQ(Pool.reserve(10, 1), 10U);
  EXPECT_EQ(Pool.reserve(10, 1), 11U);
  EXPECT_EQ(Pool.reserve(10, 1), 12U);
}

TEST(ResourcePoolTest, SeveralUnitsStartTogether) {
  ResourcePool Pool("alu", 4);
  for (int I = 0; I < 4; ++I)
    EXPECT_EQ(Pool.reserve(5, 1), 5U);
  EXPECT_EQ(Pool.reserve(5, 1), 6U);
}

TEST(ResourcePoolTest, BlockingUnitDelaysTheNextOperation) {
  ResourcePool Pool("div", 1);
  EXPECT_EQ(Pool.reserve(0, 32), 0U);
  EXPECT_EQ(Pool.reserve(1, 32), 32U);
}

TEST(ResourcePoolTest, IdleUnitStartsAtTheRequestedCycle) {
  ResourcePool Pool("lsu", 1);
  EXPECT_EQ(Pool.reserve(0, 2), 0U);
  EXPECT_EQ(Pool.reserve(100, 2), 100U);
}

TEST(ResourcePoolTest, CountsOperationsAndBusyCycles) {
  ResourcePool Pool("mul", 2);
  Pool.reserve(0, 3);
  Pool.reserve(0, 3);
  Pool.reserve(0, 1);
  EXPECT_EQ(Pool.getName(), "mul");
  EXPECT_EQ(Pool.getCount(), 2U);
  EXPECT_EQ(Pool.getOps(), 3U);
  EXPECT_EQ(Pool.getBusyCycles(), 7U);
}

TEST(ResourcePoolTest, NextFreeMatchesReserveWithoutBooking) {
  ResourcePool Pool("lsu", 1);
  Pool.reserve(0, 4);
  EXPECT_EQ(Pool.getNextFree(1), 4U);
  EXPECT_EQ(Pool.getNextFree(9), 9U);
  EXPECT_EQ(Pool.getOps(), 1U);
  EXPECT_EQ(Pool.reserve(1, 1), 4U);
}

TEST(ResourcePoolTest, ReserveFirstWaitsForTheFirstResource) {
  ResourcePool Pool("alu", 2);
  EXPECT_EQ(Pool.reserveFirst(0, 3), 0U);
  EXPECT_EQ(Pool.reserveFirst(0, 1), 3U); // The second resource stays free.
  EXPECT_EQ(Pool.reserve(0, 1), 0U);
}

TEST(ResourcePoolTest, RejectsEmptyPool) {
  EXPECT_THROW(ResourcePool("none", 0), std::invalid_argument);
}

} // namespace
