//===- VectorPipeline.h - The vector backend pipeline -----------*- C++ -*-===//
///
/// \file
/// The stages an M3 vector instruction passes through after the scalar core
/// dispatches it, and the cycle it reaches each one. Every stage is in order,
/// so an instruction's timing depends only on older instructions: schedule()
/// computes its whole path when it arrives.
///
/// Stages, with the cycle a vector add reaches each when nothing is in the
/// way (D = scalar dispatch; RTL waveform of M3, see docs/deviations.md U8):
///
///   D+1  RvvFrontEnd inst_q          (one register stage)
///   D+2  DE1 pops the command queue  (<= de1_width instructions per cycle)
///   D+3  DE2 splits into uops        (<= de2_width instructions and
///                                     <= de2_uop_width uops per cycle)
///   D+4  dispatch from the uop queue (<= dispatch_width uops per cycle)
///   D+5  the functional unit starts  (reservation station pop)
///   D+5 + latency - 1                (last cycle in the functional unit)
///   ...  result FIFO, then the ROB marks the uop done
///   D+8  the vector ROB retires it   (<= retire_width uops per cycle)
///
/// Queues between stages hold `*_entries` items. An entry freed in cycle C can
/// be refilled in cycle C + 1, because every queue's count is registered. The
/// vector ROB is the exception: dispatch reuses an entry in the cycle it
/// retires (RTL waveform of vadd_m4_indep: 2 uops retire and 2 dispatch each
/// cycle with all 8 entries in use).
///
/// Dispatch is in order: a uop waits for a vector ROB entry and an entry in
/// the reservation station of its pool, and holds back every younger uop
/// meanwhile (rvv_backend_dispatch_ctrl.sv). Each reservation station issues
/// in order to the pools it feeds. Which pools and stations exist is
/// configuration (VectorPipelineConfig), so other organizations than M3's can
/// be compared.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_VECTOR_VECTORPIPELINE_H
#define REEF_PERF_VECTOR_VECTORPIPELINE_H

#include "common/ResourcePool.h"
#include "common/Types.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace reef_perf {

/// Hands out cycles to in-order work that can do at most `Width` items per
/// cycle. Requests must come in program order.
class InOrderSlots {
public:
  /// Creates the slots for \p Width items per cycle.
  explicit InOrderSlots(uint32_t Width);

  /// \returns the cycle the next item gets: no earlier than \p Earliest or
  /// the previous item's cycle, and in a cycle with a free slot.
  Cycle take(Cycle Earliest);

private:
  uint32_t Width;
  Cycle Current = 0;
  uint32_t Used = 0;
};

/// A queue of fixed depth seen through the cycles its entries leave it.
class QueueOccupancy {
public:
  /// Creates an empty queue of \p Entries entries, whose freed entries can be
  /// refilled \p RefillDelay cycles after they leave.
  explicit QueueOccupancy(uint32_t Entries, Cycle RefillDelay = 1);

  /// \returns the first cycle the next item can enter: when the item
  /// `Entries` places ahead of it has left, plus the refill delay.
  Cycle getFreeCycle() const;

  /// Records that the next item leaves the queue in cycle \p Leave.
  void recordLeave(Cycle Leave);

private:
  uint32_t Entries;
  Cycle RefillDelay;
  std::deque<Cycle> Leaves; // The last `Entries` leave cycles.
};

/// One pool of identical functional units and the reservation station that
/// feeds it.
struct VectorPoolConfig {
  std::string Name;   ///< Name in statistics.
  uint32_t Count = 1; ///< Functional units in the pool.
  size_t Station = 0; ///< Index into VectorPipelineConfig::StationEntries.
};

/// Sizes and widths of the vector backend (rvv_backend_define.svh), and how
/// its functional units are organized.
struct VectorPipelineConfig {
  uint32_t De1Width = 2;          ///< Instructions DE1 takes per cycle.
  uint32_t LegalQueueEntries = 8; ///< Legal command queue entries.
  uint32_t De2Width = 2;          ///< Instructions DE2 takes per cycle.
  uint32_t De2UopWidth = 6;       ///< Uops DE2 writes per cycle.
  uint32_t UopQueueEntries = 16;  ///< Uop queue entries.
  uint32_t DispatchWidth = 3;     ///< Uops dispatched per cycle.
  uint32_t RobEntries = 8;        ///< Vector ROB entries (uops).
  uint32_t RetireWidth = 4;       ///< Uops retired per cycle.
  /// Entries of each reservation station; a station may feed several pools.
  std::vector<uint32_t> StationEntries;
  std::vector<VectorPoolConfig> Pools; ///< The functional unit pools.
};

/// What one instruction needs from the vector backend.
struct VectorOp {
  size_t Pool = 0;        ///< Index of the pool its uops run on.
  uint32_t Uops = 1;      ///< Uops it splits into (at least 1).
  uint64_t Latency = 1;   ///< Cycles each uop spends in its unit.
  uint64_t Occupancy = 1; ///< Cycles each uop keeps its unit busy.
  /// Only the first functional unit of the pool can run it (M3's compares).
  bool FirstUnitOnly = false;
  /// Only its last uop takes a vector ROB entry (compares, reductions).
  bool OneRobEntry = false;
};

/// When one instruction reached the pipeline's stages.
struct VectorTiming {
  Cycle CommandQueuePop = 0; ///< DE1 took it; its command-queue entry frees.
  Cycle FirstStart = 0;      ///< Its first uop entered a unit.
  Cycle Retire = 0;          ///< Its last uop retired from the vector ROB.
};

/// The vector backend's queues, decode, dispatch and ROB.
class VectorPipeline {
public:
  /// Creates an empty pipeline sized by \p Config.
  explicit VectorPipeline(const VectorPipelineConfig &Config);

  /// Schedules the next instruction in program order.
  /// \param Dispatched the cycle the scalar core dispatched it.
  /// \param Op what it needs from the backend.
  /// \param SourcesReady first cycle its uops may start, given the vector
  ///        registers it reads (Inst::VecSourcesReadyCycle).
  VectorTiming schedule(Cycle Dispatched, const VectorOp &Op,
                        Cycle SourcesReady);

  /// \returns the functional unit pools, for the end-of-run summary.
  std::vector<const ResourcePool *> getPools() const;

private:
  InOrderSlots De1Slots;
  InOrderSlots De2Slots;
  InOrderSlots De2UopSlots;
  InOrderSlots DispatchSlots;
  InOrderSlots RetireSlots;
  QueueOccupancy LegalQueue;
  QueueOccupancy UopQueue;
  QueueOccupancy Rob;
  std::vector<ResourcePool> Pools;
  std::vector<size_t> PoolStations; // Per pool: the station feeding it.
  std::vector<QueueOccupancy> Stations;
  std::vector<Cycle> LastStart; // Per station: each issues in order.
  Cycle LastUopPush = 0; // Last uop DE2 pushed into the uop queue.
  Cycle LastRetire = 0;
};

} // namespace reef_perf

#endif // REEF_PERF_VECTOR_VECTORPIPELINE_H
