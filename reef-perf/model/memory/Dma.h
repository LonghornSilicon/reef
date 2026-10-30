//===- Dma.h - The DMA initiator --------------------------------*- C++ -*-===//
///
/// \file
/// A one-channel DMA that moves data between external memory (HyperRAM) and
/// the TCMs through the core's AXI slave port, as in the Reef SoC plan
/// (atlas-soc/docs/dma.md). Not in M3: it does nothing until a program
/// writes its registers.
///
/// A program drives it with ordinary stores and loads to its register block
/// (getDmaRegsRegion()), which go out through the e-bus like any other
/// external access:
///
/// | Offset | Register | Access |
/// | --- | --- | --- |
/// | 0x0 | SRC | store: source address |
/// | 0x4 | DST | store: destination address |
/// | 0x8 | LEN | store: bytes to move |
/// | 0xc | GO | store (any value): start the transfer |
/// | 0x10 | WAIT | load: returns when every started transfer is done |
///
/// A transfer starts `dma_setup_cycles` after the GO store arrives, or after
/// the previous transfer finishes (one at a time). It then moves one
/// `dma_beat_bytes` beat per TCM port cycle the core leaves free; whichever of
/// SRC and DST is in a TCM is booked there as an AxiSlave requester. Only
/// timing is modeled: Spike's memory is not copied.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_MEMORY_DMA_H
#define REEF_PERF_MEMORY_DMA_H

#include "common/Interfaces.h"
#include "common/Types.h"

#include <cstdint>
#include <functional>

namespace reef_perf {

class Tcm;

/// Register offsets in the DMA's register block.
enum DmaRegister : uint32_t {
  DmaSrc = 0x0,
  DmaDst = 0x4,
  DmaLen = 0x8,
  DmaGo = 0xc,
  DmaWait = 0x10,
};

/// Totals for the end-of-run summary.
struct DmaStats {
  uint64_t Transfers = 0;  ///< Transfers started.
  uint64_t Bytes = 0;      ///< Bytes moved.
  uint64_t BusyCycles = 0; ///< Cycles from each start to its finish.
};

/// The DMA initiator. Owned by the memory module.
class Dma {
public:
  /// \returns the TCM holding an address, or nullptr outside the TCMs.
  using TcmLookup = std::function<Tcm *(Address)>;

  /// Creates a DMA whose transfers wait \p SetupCycles before their first
  /// beat and move \p BeatBytes per beat, booking the TCMs \p FindTcm names.
  Dma(uint32_t SetupCycles, uint32_t BeatBytes, TcmLookup FindTcm);

  /// Applies a store of \p Value to register \p Offset that arrives at
  /// cycle \p Arrive. A GO store starts a transfer.
  void writeRegister(uint32_t Offset, uint64_t Value, Cycle Arrive);

  /// \returns the cycle the last started transfer finishes, or 0 if none.
  Cycle getFinish() const { return Finish; }

  /// \returns the totals so far.
  const DmaStats &getStats() const { return Stats; }

private:
  void start(Cycle Arrive);

  uint32_t SetupCycles;
  uint32_t BeatBytes;
  TcmLookup FindTcm;
  Address Src = 0;
  Address Dst = 0;
  uint32_t Len = 0;
  Cycle Finish = 0;
  DmaStats Stats;
};

} // namespace reef_perf

#endif // REEF_PERF_MEMORY_DMA_H
