//===- Host.cpp - The external host ---------------------------------------===//

#include "memory/Host.h"

#include <cstdint>

using namespace reef_perf;

// Bytes of the write to the reset register that releases the core.
static constexpr uint64_t ReleaseWriteBytes = 4;

Host::Host(bool Enabled, uint32_t SetupCycles, uint32_t CyclesPerByte)
    : Enabled(Enabled), SetupCycles(SetupCycles), CyclesPerByte(CyclesPerByte) {
}

void Host::setBootImage(uint64_t ItcmBytes) {
  if (!Enabled)
    return;
  BootBytes = ItcmBytes + ReleaseWriteBytes;
  BootCycles = SetupCycles + (BootBytes * CyclesPerByte);
}
