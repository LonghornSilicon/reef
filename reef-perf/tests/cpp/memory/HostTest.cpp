//===- HostTest.cpp - Host boot load tests --------------------------------===//

#include "memory/Host.h"

#include <gtest/gtest.h>

using namespace reef_perf;

namespace {

TEST(HostTest, OffMeansTheCoreStartsAtOnce) {
  Host TheHost(false, true, 100, 8);
  TheHost.setBootImage(1024, 64);
  EXPECT_EQ(TheHost.getBootCycles(), 0U);
  EXPECT_EQ(TheHost.getBootBytes(), 0U);
}

TEST(HostTest, BootIsSetupPlusEachByteAndTheReleaseWrite) {
  Host TheHost(true, false, 100, 8);
  TheHost.setBootImage(88, 64); // Data off: the 64 bytes are not loaded.
  EXPECT_EQ(TheHost.getBootBytes(), 92U);          // 88 + the 4-byte release.
  EXPECT_EQ(TheHost.getBootCycles(), 100U + 736U); // 92 bytes x 8 cycles.
}

TEST(HostTest, DataBootAddsTheDtcmImageOnTheSameLink) {
  Host TheHost(true, true, 100, 8);
  TheHost.setBootImage(88, 64);
  EXPECT_EQ(TheHost.getBootDataBytes(), 64U);
  EXPECT_EQ(TheHost.getBootBytes(), 156U);          // 88 + 64 + 4.
  EXPECT_EQ(TheHost.getBootCycles(), 100U + 1248U); // 156 bytes x 8 cycles.
}

} // namespace
