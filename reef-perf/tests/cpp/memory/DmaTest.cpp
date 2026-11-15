//===- DmaTest.cpp - DMA initiator tests ----------------------------------===//

#include "memory/Dma.h"
#include "memory/MemTiming.h"
#include "memory/Tcm.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace reef_perf;

namespace {

/// A DTCM at 0x10000 (32 KB) with a 1-cycle SRAM, and a DMA that books it.
struct DmaFixture : public ::testing::Test {
  Tcm DTCM{"dtcm", 1, 16, 1};
  Dma DMA{20, 16, nullptr, [this](Address Addr) -> Tcm * {
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
  Dma Narrow{20, 4, nullptr,
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
  Dma Wide{20, 32, nullptr,
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

/// The same DTCM with a modeled external link: 26 cycles to the first byte,
/// then 1 B/cycle (a 16-bit QSPI-class link at half the core clock).
struct DmaLinkFixture : public DmaFixture {
  ExtMemModel Ext{26, 1.0};
  Dma LinkDMA{20, 16, [this](Address) { return &Ext; },
              [this](Address Addr) -> Tcm * {
                return Addr >= 0x10000 && Addr < 0x18000 ? &DTCM : nullptr;
              }};

  void programLink(Address Src, Address Dst, uint32_t Len, Cycle At) {
    LinkDMA.writeRegister(DmaSrc, Src, At);
    LinkDMA.writeRegister(DmaDst, Dst, At);
    LinkDMA.writeRegister(DmaLen, Len, At);
    LinkDMA.writeRegister(DmaGo, 1, At);
  }
};

TEST_F(DmaLinkFixture, ExternalEndIsBoundByTheLink) {
  // Begin 120; the first byte at 146, then 16 bytes every 16 cycles. Each
  // DTCM beat waits for its bytes: 162, 178, 194 and 210, usable at 211.
  programLink(0x20000000, 0x10000, 64, 100);
  EXPECT_EQ(LinkDMA.getFinish(), 211U);
  EXPECT_EQ(Ext.getLink().getBusyCycles(), 90U);
}

TEST_F(DmaLinkFixture, CoreAccessWhileBytesArriveBumpsABeat) {
  programLink(0x20000000, 0x10000, 64, 100); // Beats 162, 178, 194, 210.
  std::vector<MemAccess> Load = {{0x10100, 4, false, 0}};
  MemRequest Request;
  Request.Accesses = Load;
  Request.Earliest = 178; // The second beat's cycle.
  EXPECT_EQ(DTCM.access(Request).Start, 178U);
  // The beat takes the free cycle after it, so the finish holds.
  EXPECT_EQ(DTCM.getStats(RequesterKind::AxiSlave).BumpedCycles, 1U);
  EXPECT_EQ(LinkDMA.getFinish(), 211U);
  Request.Earliest = 210; // The last beat's cycle.
  DTCM.access(Request);
  EXPECT_EQ(LinkDMA.getFinish(), 212U);
}

TEST_F(DmaLinkFixture, QueuedTransferWaitsForTheLink) {
  programLink(0x20000000, 0x10000, 32, 0); // Link 20-78, beats 62 and 78.
  programLink(0x20000000, 0x10100, 32, 5); // Begin 99, link 99-157.
  EXPECT_EQ(LinkDMA.getFinish(), 158U);    // Last beat at 157.
}

TEST_F(DmaLinkFixture, TcmToTcmDoesNotUseTheLink) {
  programLink(0x10000, 0x10100, 64, 0);
  EXPECT_EQ(Ext.getLink().getOps(), 0U);
}

/// The same DTCM with a DMA that holds at most \p Depth unfinished transfers.
struct DmaQueueFixture : public DmaFixture {
  Dma Queued{20, 16, nullptr,
             [this](Address Addr) -> Tcm * {
               return Addr >= 0x10000 && Addr < 0x18000 ? &DTCM : nullptr;
             },
             1};

  Cycle go(Address Dst, uint32_t Len, Cycle At) {
    Queued.writeRegister(DmaSrc, 0x20000000, At);
    Queued.writeRegister(DmaDst, Dst, At);
    Queued.writeRegister(DmaLen, Len, At);
    return Queued.writeRegister(DmaGo, 1, At);
  }
};

TEST_F(DmaQueueFixture, GoWithRoomIsTakenAtOnce) {
  EXPECT_EQ(go(0x10000, 32, 0), 0U); // Beats 20-21, done at 22.
  EXPECT_EQ(Queued.getStats().FullCycles, 0U);
}

TEST_F(DmaQueueFixture, GoIntoAFullQueueWaitsForTheOldestToFinish) {
  go(0x10000, 32, 0);                 // Done at 22.
  EXPECT_EQ(go(0x10100, 32, 5), 22U); // Waits 17 cycles for a slot.
  EXPECT_EQ(Queued.getFinish(), 44U); // 22 + 20 setup + 2 beats.
  EXPECT_EQ(Queued.getStats().FullCycles, 17U);
}

TEST_F(DmaQueueFixture, GoAfterTheQueueDrainsDoesNotWait) {
  go(0x10000, 32, 0);
  EXPECT_EQ(go(0x10100, 32, 30), 30U);
}

TEST_F(DmaFixture, WithoutADepthTheQueueNeverFills) {
  program(0x80000000, 0x10000, 32, 0);
  DMA.writeRegister(DmaDst, 0x10100, 5);
  EXPECT_EQ(DMA.writeRegister(DmaGo, 1, 5), 5U);
}

/// A flash link and a slower RAM link from 0x24000000 up; the DMA uses the
/// link its external address is behind.
struct DmaTwoLinkFixture : public DmaFixture {
  ExtMemModel Flash{"flash", 26, 26, 1.0, 0};
  ExtMemModel Ram{"ram", 30, 16, 0.25, 0};
  Dma TwoDMA{
      20, 16,
      [this](Address Addr) { return Addr >= 0x24000000 ? &Ram : &Flash; },
      [this](Address Addr) -> Tcm * {
        return Addr >= 0x10000 && Addr < 0x18000 ? &DTCM : nullptr;
      }};

  void programTwo(Address Src, Address Dst, uint32_t Len, Cycle At) {
    TwoDMA.writeRegister(DmaSrc, Src, At);
    TwoDMA.writeRegister(DmaDst, Dst, At);
    TwoDMA.writeRegister(DmaLen, Len, At);
    TwoDMA.writeRegister(DmaGo, 1, At);
  }
};

TEST_F(DmaTwoLinkFixture, ARamAddressUsesTheRamLink) {
  // Begin 20; the RAM's first byte at 50, then 16 bytes at 0.25 B/cycle: the
  // DTCM beat at 114, usable at 115.
  programTwo(0x24000000, 0x10000, 16, 0);
  EXPECT_EQ(TwoDMA.getFinish(), 115U);
  EXPECT_EQ(Ram.getLink().getBusyCycles(), 94U);
  EXPECT_EQ(Flash.getLink().getOps(), 0U);
}

TEST_F(DmaTwoLinkFixture, AWriteToTheRamPaysTheWriteLatency) {
  // Begin 20; the RAM write takes 16 + 64 cycles, to 100.
  programTwo(0x10000, 0x24000000, 16, 0);
  EXPECT_EQ(TwoDMA.getFinish(), 100U);
  EXPECT_EQ(Ram.getLink().getBusyCycles(), 80U);
}

TEST_F(DmaTwoLinkFixture, TheTwoLinksDoNotWaitForEachOther) {
  // A core read holds the flash link; the RAM link is still free.
  Flash.transfer(1000, 0);
  programTwo(0x24000000, 0x10000, 16, 0);
  EXPECT_EQ(TwoDMA.getFinish(), 115U);
}

} // namespace
