//===- Types.h - Purpose-defined types shared by every module ---*- C++ -*-===//
///
/// \file
/// Names for the values that cross module boundaries, so a signature says
/// what it takes. Cycle and Address are aliases: they document intent but
/// convert freely, because Sparta's own APIs use plain integers. RegId is a
/// real type, because a register has structure (a file and an index) that a
/// bare integer hides.
///
/// Groups of instructions (fetch packets, dispatch groups) get their own
/// types when the frontend and core model them (roadmap B06-B08).
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_COMMON_TYPES_H
#define REEF_PERF_COMMON_TYPES_H

#include <compare>
#include <cstdint>
#include <ostream>

namespace reef_perf {

/// A point in simulated time, in core clock cycles. Durations (latencies,
/// occupancies) stay plain integers.
using Cycle = uint64_t;

/// A byte address in the core's 32-bit physical address space.
using Address = uint32_t;

/// An architectural register file.
enum class RegFile : uint8_t {
  X, ///< Integer registers x0-x31.
  F, ///< Floating-point registers f0-f31.
  V  ///< Vector registers v0-v31.
};

/// An architectural register: a register file and an index into it.
class RegId {
public:
  static constexpr unsigned NumPerFile = 32; ///< Registers in each file.
  static constexpr unsigned NumDense = 3 * NumPerFile; ///< See getDense().

  /// Creates register \p Index of \p File; \p Index must be below 32.
  constexpr RegId(RegFile File, unsigned Index)
      : File(File), Index(static_cast<uint8_t>(Index)) {}

  /// \returns integer register x\p Index.
  static constexpr RegId x(unsigned Index) { return {RegFile::X, Index}; }
  /// \returns floating-point register f\p Index.
  static constexpr RegId f(unsigned Index) { return {RegFile::F, Index}; }
  /// \returns vector register v\p Index.
  static constexpr RegId v(unsigned Index) { return {RegFile::V, Index}; }

  /// \returns the register file.
  constexpr RegFile getFile() const { return File; }
  /// \returns the index within the file.
  constexpr unsigned getIndex() const { return Index; }

  /// \returns a number below NumDense, unique across files, for arrays
  /// indexed by register (x0-x31, then f0-f31, then v0-v31).
  constexpr unsigned getDense() const {
    return static_cast<unsigned>(File) * NumPerFile + Index;
  }

  /// Orders by file, then index: the same order as getDense().
  friend constexpr auto operator<=>(const RegId &, const RegId &) = default;

private:
  RegFile File;
  uint8_t Index;
};

/// Writes \p Reg as its assembly name, e.g. "x10" or "v4".
inline std::ostream &operator<<(std::ostream &OS, RegId Reg) {
  return OS << "xfv"[static_cast<unsigned>(Reg.getFile())] << Reg.getIndex();
}

} // namespace reef_perf

#endif // REEF_PERF_COMMON_TYPES_H
