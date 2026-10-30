//===- HostTest.cpp - Host boot load tests --------------------------------===//

#include "memory/Host.h"

#include <gtest/gtest.h>

using namespace reef_perf;

namespace {

TEST(HostTest, OffMeansTheCoreStartsAtOnce) {
  Host TheHost(false, 100, 8);
  TheHost.setBootImage(1024);
  EXPECT_EQ(TheHost.getBootCycles(), 0U);
  EXPECT_EQ(TheHost.getBootBytes(), 0U);
}

TEST(HostTest, BootIsSetupPlusEachByteAndTheReleaseWrite) {
  Host TheHost(true, 100, 8);
  TheHost.setBootImage(88);
  EXPECT_EQ(TheHost.getBootBytes(), 92U);          // 88 + the 4-byte release.
  EXPECT_EQ(TheHost.getBootCycles(), 100U + 736U); // 92 bytes x 8 cycles.
}

} // namespace
