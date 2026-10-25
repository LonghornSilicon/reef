//===- InstTest.cpp - Instruction helper tests ----------------------------===//

#include "common/Inst.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace reef_perf;

/// \returns accesses of \p Count consecutive \p Size-byte elements from
/// \p Base, in element order.
static std::vector<MemAccess> makeElements(Address Base, uint32_t Count,
                                           uint8_t Size) {
  std::vector<MemAccess> Out;
  Out.reserve(Count);
  for (uint32_t Idx = 0; Idx < Count; ++Idx)
    Out.push_back({Base + (Idx * Size), Size, false});
  return Out;
}

namespace {

TEST(SplitByRegisterTest, OneRegisterIsOneGroup) {
  auto Accesses = makeElements(0x10000, 16, 1);
  auto Groups = splitByRegister(Accesses, 16);
  ASSERT_EQ(Groups.size(), 1U);
  EXPECT_EQ(Groups[0].size(), 16U);
}

TEST(SplitByRegisterTest, RegisterGroupSplitsIntoOneGroupPerRegister) {
  auto Accesses = makeElements(0x10000, 64, 1); // vle8 at LMUL = 4
  auto Groups = splitByRegister(Accesses, 16);
  ASSERT_EQ(Groups.size(), 4U);
  for (const auto &Group : Groups)
    EXPECT_EQ(Group.size(), 16U);
  EXPECT_EQ(Groups[1].front().Addr, 0x10010U);
}

TEST(SplitByRegisterTest, WiderElementsFillARegisterSooner) {
  auto Accesses = makeElements(0x10000, 8, 4); // vle32, 32 bytes
  auto Groups = splitByRegister(Accesses, 16);
  ASSERT_EQ(Groups.size(), 2U);
  EXPECT_EQ(Groups[0].size(), 4U);
}

TEST(SplitByRegisterTest, PartlyFilledLastRegisterIsItsOwnGroup) {
  auto Accesses = makeElements(0x10000, 5, 4); // vl = 5, SEW = 32
  auto Groups = splitByRegister(Accesses, 16);
  ASSERT_EQ(Groups.size(), 2U);
  EXPECT_EQ(Groups[1].size(), 1U);
}

TEST(SplitByRegisterTest, NoAccessesIsNoGroups) {
  EXPECT_TRUE(splitByRegister({}, 16).empty());
}

TEST(SplitByRegisterTest, ZeroRegisterSizeThrows) {
  auto Accesses = makeElements(0x10000, 4, 4);
  EXPECT_THROW(splitByRegister(Accesses, 0), std::invalid_argument);
}

} // namespace
