//===- Vector.cpp - The vector module -------------------------------------===//

#include "vector/Vector.h"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace reef_perf;

/// \returns the mnemonic up to its first '.', e.g. "vwredsum".
static std::string_view getStem(const Inst &I) {
  std::string_view Mnemonic = I.Mnemonic;
  return Mnemonic.substr(0, Mnemonic.find('.'));
}

static bool startsWithAny(std::string_view S,
                          std::initializer_list<std::string_view> Prefixes) {
  return std::ranges::any_of(
      Prefixes, [S](std::string_view P) { return S.starts_with(P); });
}

static bool isReduction(const Inst &I) {
  return startsWithAny(getStem(I), {"vred", "vwred", "vfred", "vfwred"});
}

/// \returns true for the integer compares, which only the first ALU runs
/// (rvv_backend_alu.sv) and which write one mask register.
static bool isIntegerCompare(const Inst &I) {
  static const std::set<std::string_view> Stems = {
      "vmseq", "vmsne",  "vmsltu", "vmslt", "vmsleu",
      "vmsle", "vmsgtu", "vmsgt",  "vmadc", "vmsbc"};
  return Stems.count(getStem(I)) != 0;
}

static bool isFpCompare(const Inst &I) {
  return startsWithAny(getStem(I),
                       {"vmfeq", "vmfne", "vmflt", "vmfle", "vmfgt", "vmfge"});
}

/// \returns true for widening and narrowing ops, whose widest operand spans
/// 2 x LMUL registers.
static bool usesDoubleGroup(const Inst &I) {
  return !isReduction(I) &&
         startsWithAny(getStem(I), {"vw", "vfw", "vnsr", "vnclip", "vfncvt"});
}

uint32_t reef_perf::getVectorUops(const Inst &I) {
  std::string_view Mnemonic = I.Mnemonic;
  std::string_view Stem = getStem(I);
  // Scalar moves, mask-register ops and mask-to-scalar ops use one register.
  if (startsWithAny(Mnemonic, {"vmv.s.x", "vmv.x.s", "vfmv.s.f", "vfmv.f.s",
                               "vcpop", "vfirst", "vmsbf", "vmsif", "vmsof"}) ||
      startsWithAny(Stem,
                    {"vmand", "vmnand", "vmor", "vmnor", "vmxor", "vmxnor"}))
    return 1;
  // vmv<N>r.v moves N whole registers.
  if (Mnemonic.size() > 4 && Mnemonic.starts_with("vmv") && Mnemonic[4] == 'r')
    return static_cast<uint32_t>(Mnemonic[3] - '0');
  uint32_t EMUL8 = I.LMUL8;
  // Widening and narrowing ops read or write 2 x LMUL registers; a reduction
  // walks its LMUL source registers into one result.
  if (usesDoubleGroup(I))
    EMUL8 *= 2;
  return std::clamp<uint32_t>(EMUL8 / 8, 1, 8);
}

VectorGroup reef_perf::getVectorGroup(const Inst &I) {
  std::string_view Stem = getStem(I);
  if (startsWithAny(Stem, {"vfred", "vfwred"}))
    return VectorGroup::FpReduce;
  if (Stem.starts_with("vwred"))
    return VectorGroup::WideningReduce;
  if (startsWithAny(Stem, {"vred", "vcpop", "viota"}))
    return VectorGroup::Reduce;
  switch (I.Class) {
  case InstClass::V_PERM:
    return VectorGroup::Permute;
  case InstClass::V_MUL:
    return VectorGroup::Mul;
  case InstClass::V_DIV:
    return VectorGroup::Div;
  case InstClass::V_FDIV:
    return VectorGroup::FDiv;
  case InstClass::V_FP:
    return startsWithAny(Stem, {"vfmv", "vfmerge"}) ? VectorGroup::Alu
                                                    : VectorGroup::Fp;
  default:
    if (isIntegerCompare(I))
      return VectorGroup::Compare;
    return usesAddSubStage(I) ? VectorGroup::AluAddSub : VectorGroup::Alu;
  }
}

bool reef_perf::usesAddSubStage(const Inst &I) {
  static const std::set<std::string_view> Stems = {
      "vadd",   "vsub",   "vrsub",  "vadc",   "vsbc",   "vmadc",  "vmsbc",
      "vsadd",  "vsaddu", "vssub",  "vssubu", "vaadd",  "vaaddu", "vasub",
      "vasubu", "vwadd",  "vwaddu", "vwsub",  "vwsubu", "vmin",   "vminu",
      "vmax",   "vmaxu",  "vmseq",  "vmsne",  "vmslt",  "vmsltu", "vmsle",
      "vmsleu", "vmsgt",  "vmsgtu"};
  return Stems.count(getStem(I)) != 0;
}

/// \returns the index of \p Name in \p Names.
/// \throws std::invalid_argument naming \p What if it isn't there.
static size_t findName(const std::vector<std::string> &Names,
                       const std::string &Name, const std::string &What) {
  auto It = std::ranges::find(Names, Name);
  if (It == Names.end())
    throw std::invalid_argument(What + " '" + Name + "' is not defined");
  return static_cast<size_t>(It - Names.begin());
}

/// \throws std::invalid_argument if \p Names has a duplicate.
static void checkUnique(const std::vector<std::string> &Names,
                        const std::string &What) {
  for (size_t I = 0; I < Names.size(); ++I)
    if (std::ranges::count(Names, Names[I]) > 1)
      throw std::invalid_argument(What + " '" + Names[I] +
                                  "' is defined twice");
}

static VectorPipelineConfig
getPipelineConfig(const Vector::VectorParameterSet *Params) {
  VectorPipelineConfig Config;
  Config.De1Width = Params->vec_de1_width;
  Config.LegalQueueEntries = Params->vec_legal_queue_entries;
  Config.De2Width = Params->vec_de2_width;
  Config.De2UopWidth = Params->vec_de2_uop_width;
  Config.UopQueueEntries = Params->vec_uop_queue_entries;
  Config.DispatchWidth = Params->vec_dispatch_width;
  Config.RobEntries = Params->vec_rob_entries;
  Config.RetireWidth = Params->vec_retire_width;

  const std::vector<std::string> &Stations = Params->vec_stations;
  const std::vector<std::string> &Pools = Params->vec_pools;
  const std::vector<uint32_t> &Counts = Params->vec_pool_counts;
  const std::vector<std::string> &PoolStations = Params->vec_pool_stations;
  checkUnique(Stations, "vector station");
  checkUnique(Pools, "vector pool");
  if (Params->vec_station_entries.getValue().size() != Stations.size())
    throw std::invalid_argument(
        "vec_station_entries needs one value per vec_stations name");
  if (Counts.size() != Pools.size() || PoolStations.size() != Pools.size())
    throw std::invalid_argument("vec_pool_counts and vec_pool_stations need "
                                "one value per vec_pools name");
  Config.StationEntries = Params->vec_station_entries;
  Config.LsuStationEntries = Params->vec_lsu_station_entries;
  for (size_t I = 0; I < Pools.size(); ++I)
    Config.Pools.push_back({Pools[I], Counts[I],
                            findName(Stations, PoolStations[I],
                                     "vec_pool_stations: vector station")});
  return Config;
}

/// \returns the pool index of each instruction group.
static std::array<size_t, NumVectorGroups>
getGroupPools(const Vector::VectorParameterSet *Params) {
  // In VectorGroup order.
  const std::array<const sparta::Parameter<std::string> *, NumVectorGroups>
      Names = {&Params->vec_alu_addsub_pool, &Params->vec_alu_pool,
               &Params->vec_compare_pool,    &Params->vec_mul_pool,
               &Params->vec_div_pool,        &Params->vec_fdiv_pool,
               &Params->vec_fp_pool,         &Params->vec_permute_pool,
               &Params->vec_reduce_pool,     &Params->vec_widening_reduce_pool,
               &Params->vec_fp_reduce_pool};
  std::array<size_t, NumVectorGroups> GroupPools{};
  for (size_t G = 0; G < NumVectorGroups; ++G)
    GroupPools[G] = findName(Params->vec_pools, *Names[G],
                             Names[G]->getName() + ": vector pool");
  return GroupPools;
}

Vector::Vector(sparta::TreeNode *Node, const VectorParameterSet *Params)
    : sparta::Unit(Node, name), QueueEntries(Params->vec_queue_entries),
      AddSubLatency(Params->vec_addsub_latency),
      AluLatency(Params->vec_alu_latency), MulLatency(Params->vec_mul_latency),
      DivCyclesPerUop(Params->vec_div_cycles_per_uop),
      FDivLatency(Params->vec_fdiv_latency),
      FDivCyclesPerUop(Params->vec_fdiv_cycles_per_uop),
      FpLatency(Params->vec_fp_latency),
      PermuteLatency(Params->vec_permute_latency),
      PermuteCyclesPerUop(Params->vec_permute_cycles_per_uop),
      ReduceLatency(Params->vec_reduce_latency),
      WideningReduceCyclesPerUop(Params->vec_widening_reduce_cycles_per_uop),
      FpReduceLatency(Params->vec_fp_reduce_latency),
      FpReduceCyclesPerUop(Params->vec_fp_reduce_cycles_per_uop),
      ToScalarLatency(Params->vec_to_scalar_latency),
      VsetLatency(Params->vset_latency),
      CompareFirstUnitOnly(Params->vec_compare_first_unit_only),
      GroupPools(getGroupPools(Params)), Pipeline(getPipelineConfig(Params)) {
  InInstsPort.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Vector, receiveInst, InstRef));
  sparta::StartupEvent(Node, CREATE_SPARTA_HANDLER(Vector, sendInitialCredits));
}

void Vector::sendInitialCredits() { OutCreditsPort.send(QueueEntries); }

VectorOp Vector::getOp(const Inst &I) const {
  VectorGroup Group = getVectorGroup(I);
  VectorOp Op;
  Op.Pool = GroupPools[static_cast<size_t>(Group)];
  Op.Uops = getVectorUops(I);
  Op.FirstUnitOnly = Group == VectorGroup::Compare && CompareFirstUnitOnly;
  Op.OneRobEntry = isIntegerCompare(I) || isFpCompare(I) || isReduction(I);
  switch (Group) {
  case VectorGroup::AluAddSub:
  case VectorGroup::Compare:
    Op.Latency = AddSubLatency;
    break;
  case VectorGroup::Alu:
    Op.Latency = AluLatency;
    break;
  case VectorGroup::Mul:
    Op.Latency = MulLatency;
    break;
  case VectorGroup::Div:
    // The divider holds a uop until its result leaves.
    Op.Latency = DivCyclesPerUop;
    Op.Occupancy = DivCyclesPerUop;
    break;
  case VectorGroup::FDiv:
    Op.Latency = FDivLatency;
    Op.Occupancy = FDivCyclesPerUop;
    break;
  case VectorGroup::Fp:
    Op.Latency = FpLatency;
    break;
  case VectorGroup::Permute:
    Op.Latency = PermuteLatency;
    Op.Occupancy = PermuteCyclesPerUop;
    break;
  case VectorGroup::Reduce:
    Op.Latency = ReduceLatency;
    break;
  case VectorGroup::WideningReduce:
    // Each uop goes through the reduction ALU in two halves
    // (rvv_backend_pmtrdt_unit_reduction.sv, wsum_h).
    Op.Occupancy = WideningReduceCyclesPerUop;
    Op.Latency = ReduceLatency + WideningReduceCyclesPerUop - 1;
    break;
  case VectorGroup::FpReduce:
    Op.Latency = FpReduceLatency;
    Op.Occupancy = FpReduceCyclesPerUop;
    break;
  }
  return Op;
}

std::vector<Cycle> Vector::getSourcesReady(const Inst &I, uint32_t Uops) const {
  std::vector<unsigned> Regs;
  for (RegId Reg : I.Srcs)
    if (Reg.getFile() == RegFile::V)
      Regs.push_back(Reg.getIndex());
  if (Regs.empty())
    return {};

  // Srcs lists each operand's register group in turn. A group is aligned to
  // its size: LMUL registers, 2 x LMUL for a widening or narrowing operand,
  // or one (v0, a reduction's vs1).
  uint32_t EMUL = std::max<uint32_t>(1, I.LMUL8 / 8);
  std::vector<uint32_t> Sizes = {EMUL};
  if (usesDoubleGroup(I))
    Sizes.insert(Sizes.begin(), 2 * EMUL);
  std::vector<Cycle> Ready(Uops, 0);
  for (size_t Begin = 0; Begin < Regs.size();) {
    uint32_t Size = 1;
    for (uint32_t Try : Sizes) {
      bool Fits = Regs[Begin] % Try == 0 && Begin + Try <= Regs.size();
      for (uint32_t K = 1; Fits && K < Try; ++K)
        Fits = Regs[Begin + K] == Regs[Begin] + K;
      if (Fits) {
        Size = Try;
        break;
      }
    }
    // Uop U reads the part of the group in the same position, as M3 does
    // (one register of an LMUL group, half of one for a widening op).
    for (uint32_t Uop = 0; Uop < Uops; ++Uop) {
      unsigned Reg =
          Regs[Begin + std::min<size_t>(Size - 1, size_t{Uop} * Size / Uops)];
      Ready[Uop] = std::max(Ready[Uop], RegReady[Reg]);
    }
    Begin += Size;
  }
  return Ready;
}

void Vector::recordResults(const Inst &I, const VectorTiming &T) {
  std::vector<unsigned> Regs;
  for (RegId Reg : I.Dsts)
    if (Reg.getFile() == RegFile::V)
      Regs.push_back(Reg.getIndex());
  // Register K of the destination group is ready once the last uop writing it
  // retires (two uops each for a narrowing op).
  size_t Uops = T.UopRetires.size();
  for (size_t K = 0; K < Regs.size(); ++K) {
    size_t Writer = std::min(Uops, (K + 1) * Uops / Regs.size());
    RegReady[Regs[K]] = T.UopRetires[Writer > 0 ? Writer - 1 : 0] + 1;
  }
}

void Vector::receiveInst(const InstRef &I) {
  Cycle Now = getClock()->currentCycle();
  ++NumExecuted;

  // vset* is resolved in the frontend and never reaches the command queue;
  // its credit returns when it leaves the frontend.
  if (I->Class == InstClass::VSET) {
    I->IssueCycle = Now;
    I->ResultReadyCycle = Now + VsetLatency - 1;
    I->CompleteCycle = Now + VsetLatency;
    OutCreditsPort.send(1);
    return;
  }

  VectorOp Op;
  // A load or store is one uop per vector register, each run by the core's
  // LSU on the accesses Spike recorded for it.
  std::vector<std::span<const MemAccess>> Registers;
  if (I->Class == InstClass::V_LOAD || I->Class == InstClass::V_STORE) {
    sparta_assert(Lsu, "the core's LSU was not connected to the vector module");
    Registers = splitByRegister(I->Mem, VectorRegisterBytes);
    if (Registers.empty())
      Registers.emplace_back(I->Mem);
    Op.Uops = static_cast<uint32_t>(Registers.size());
    Op.BookLsu = [&](uint32_t Uop, Cycle Earliest) {
      VectorLsuRequest Request;
      Request.IsStore = I->Class == InstClass::V_STORE;
      Request.Accesses = Registers.at(Uop);
      Request.Earliest = Earliest;
      Request.LastRegister = Uop + 1 == Registers.size();
      return Lsu->accessVectorRegister(Request);
    };
  } else {
    Op = getOp(*I);
  }
  VectorTiming T =
      Pipeline.schedule(I->DispatchCycle, Op, getSourcesReady(*I, Op.Uops));
  recordResults(*I, T);
  NumUops += Op.Uops;

  I->IssueCycle = T.FirstStart;
  I->ResultReadyCycle = T.Retire;
  if (I->Class == InstClass::V_TO_SCALAR)
    I->ResultReadyCycle += ToScalarLatency;
  I->CompleteCycle = T.Retire;
  // The scalar retirement buffer marks a store done a cycle after the LSU's
  // first store-complete signal for its PC, which its first register sends
  // (RetirementBuffer.scala, storeComplete).
  if (I->Class == InstClass::V_STORE)
    I->CompleteCycle = T.FirstDone + 1;

  // Dispatch may use the command-queue entry the cycle after DE1 takes it.
  sparta_assert(T.CommandQueuePop >= Now);
  OutCreditsPort.send(1, T.CommandQueuePop - Now);
}
