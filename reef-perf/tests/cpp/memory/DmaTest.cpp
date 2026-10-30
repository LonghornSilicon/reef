//===- DmaTest.cpp - DMA initiator tests ----------------------------------===//

#include "memory/Dma.h"
#include "memory/Tcm.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace reef_perf;

namespace {

/// A DTCM at 0x10000 (32 KB) with a 1-cycle SRAM, and a DMA that books it.
struct DmaFixture : public ::testing::Test {
  Tcm DTCM{"dtcm", 1, 16, 1};
  Dma DMA{20, 16, [this](Address Addr) -> Tcm * {
            return Addr >= 0x10000 && Addr < 0x18000 ? &DTCM : nullptr;
          }};

  /// Programs a transfer of \p Len bytes from \p Src to \p Dst, GO at \p At.
  void program(Address Src, Address Dst, uint32_t Len, Cycle At) {
    DMA.writeRegister(DmaSrc, Src, At);
    DMA.writeRegister(DmaDst, Dst, At);
    DMA.writeRegister(DmaLen, Len, At);
    DMA.writeRegister(DmaGo, 1, At);
  }
};

TEST_F(DmaFixture, NothingHappensWithoutGo) {
  DMA.writeRegister(DmaLen, 64, 0);
  EXPECT_EQ(DMA.getFinish(), 0U);
  EXPECT_EQ(DMA.getStats().Transfers, 0U);
}

TEST_F(DmaFixture, ExternalToDtcmWaitsForSetupThenOneBeatPerCycle) {
  // 64 bytes = 4 beats from HyperRAM into the DTCM, GO at cycle 100: beats
  // in 120-123, the last usable a cycle later.
  program(0x80000000, 0x10000, 64, 100);
  EXPECT_EQ(DMA.getFinish(), 124U);
  EXPECT_EQ(DTCM.getStats(RequesterKind::AxiSlave).Writes, 1U);
  EXPECT_EQ(DTCM.getStats(RequesterKind::AxiSlave).PortCycles, 4U);
  EXPECT_EQ(DMA.getStats().Bytes, 64U);
}

TEST_F(DmaFixture, CoreAccessDuringTheTransferPushesItsFinishOut) {
  program(0x80000000, 0x10000, 64, 100);
  std::vector<MemAccess> Load = {{0x10100, 4, false, 0}};
  MemRequest Request;
  Request.Accesses = Load;
  Request.Earliest = 121;
  EXPECT_EQ(DTCM.access(Request).Start, 121U); // The core never waits.
  EXPECT_EQ(DMA.getFinish(), 125U);            // One beat moved to 124.
}

TEST_F(DmaFixture, SecondTransferStartsAfterTheFirst) {
  program(0x80000000, 0x10000, 32, 0); // Beats 20-21, done at 22.
  program(0x80000000, 0x10100, 32, 5); // Waits for 22, then 20 setup.
  EXPECT_EQ(DMA.getFinish(), 44U);     // Beats 42-43.
}

TEST_F(DmaFixture, ZeroLengthDoesNothing) {
  program(0x80000000, 0x10000, 0, 0);
  EXPECT_EQ(DMA.getStats().Transfers, 0U);
}

} // namespace
