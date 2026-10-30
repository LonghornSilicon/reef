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

TEST_F(DmaFixture, QueuedTransferMovesWhenTheOneAheadIsPushedOut) {
  program(0x80000000, 0x10000, 64, 0); // Beats 20-23, done at 24.
  program(0x80000000, 0x10100, 32, 0); // Beats 44-45, done at 46.
  EXPECT_EQ(DMA.getFinish(), 46U);
  // The core takes 20-23: the first transfer moves to 24-27 and finishes at
  // 28, so the second starts at 48 and finishes at 50.
  std::vector<MemAccess> Load = {{0x10200, 16, false, 0}};
  MemRequest Request;
  Request.Accesses = Load;
  for (Cycle C = 20; C < 24; ++C) {
    Request.Earliest = C;
    DTCM.access(Request);
  }
  EXPECT_EQ(DMA.getFinish(), 50U);
  EXPECT_EQ(DMA.getStats().BusyCycles, 10U); // 20-28 and 48-50.
}

TEST_F(DmaFixture, NarrowBeatsTakeOnePortCycleEach) {
  Tcm Port{"dtcm", 1, 16, 1};
  Dma Narrow{20, 4,
             [&Port](Address A) { return A < 0x80000000 ? &Port : nullptr; }};
  Narrow.writeRegister(DmaSrc, 0x80000000, 0);
  Narrow.writeRegister(DmaDst, 0x10000, 0);
  Narrow.writeRegister(DmaLen, 16, 0);
  Narrow.writeRegister(DmaGo, 1, 0);
  EXPECT_EQ(Port.getStats(RequesterKind::AxiSlave).PortCycles, 4U);
  EXPECT_EQ(Narrow.getFinish(), 24U); // Beats 20-23.
}

TEST_F(DmaFixture, LastBeatCarriesOnlyTheBytesLeft) {
  Tcm Port{"dtcm", 1, 16, 1};
  Dma Wide{20, 32,
           [&Port](Address A) { return A < 0x80000000 ? &Port : nullptr; }};
  Wide.writeRegister(DmaSrc, 0x80000000, 0);
  Wide.writeRegister(DmaDst, 0x10000, 0);
  Wide.writeRegister(DmaLen, 48, 0);
  Wide.writeRegister(DmaGo, 1, 0);
  // One 32-byte beat (2 lines) and one 16-byte beat (1 line).
  EXPECT_EQ(Port.getStats(RequesterKind::AxiSlave).PortCycles, 3U);
  EXPECT_EQ(Port.getStats(RequesterKind::AxiSlave).Bytes, 48U);
}

TEST_F(DmaFixture, ZeroLengthDoesNothing) {
  program(0x80000000, 0x10000, 0, 0);
  EXPECT_EQ(DMA.getStats().Transfers, 0U);
}

} // namespace
