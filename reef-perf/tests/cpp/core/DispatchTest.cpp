//===- DispatchTest.cpp - Dispatch tests ----------------------------------===//

#include "core/Dispatch.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>

using namespace reef_perf;

/// \returns an in-flight instruction whose result is ready at \p Ready.
static InstRef makeWriter(Cycle Ready) {
  auto I = std::make_shared<Inst>();
  I->ResultReadyCycle = Ready;
  return I;
}

namespace {

using Scoreboard = std::array<InstRef, RegId::NumDense>;

TEST(VecSourcesReadyTest, StartsTheCycleAfterTheLatestSource) {
  Scoreboard LastWriter{};
  LastWriter.at(RegId::v(1).getDense()) = makeWriter(10);
  LastWriter.at(RegId::v(2).getDense()) = makeWriter(25);
  Inst I;
  I.Srcs = {RegId::v(1), RegId::v(2)};
  EXPECT_EQ(getVecSourcesReady(I, LastWriter), 26U);
}

TEST(VecSourcesReadyTest, ScalarSourcesAreIgnored) {
  Scoreboard LastWriter{};
  LastWriter.at(RegId::x(5).getDense()) = makeWriter(40); // A store's base.
  LastWriter.at(RegId::v(3).getDense()) = makeWriter(12);
  Inst I;
  I.Srcs = {RegId::x(5), RegId::v(3)};
  EXPECT_EQ(getVecSourcesReady(I, LastWriter), 13U);
}

TEST(VecSourcesReadyTest, NoWriterIsReadyAtOnce) {
  const Scoreboard LastWriter{};
  Inst I;
  I.Srcs = {RegId::v(8)};
  EXPECT_EQ(getVecSourcesReady(I, LastWriter), 0U);
}

TEST(VecSourcesReadyTest, UnscheduledWriterIsNeverReady) {
  Scoreboard LastWriter{};
  LastWriter.at(RegId::v(4).getDense()) = makeWriter(Inst::Never);
  Inst I;
  I.Srcs = {RegId::v(4)};
  EXPECT_EQ(getVecSourcesReady(I, LastWriter), Inst::Never);
}

} // namespace
