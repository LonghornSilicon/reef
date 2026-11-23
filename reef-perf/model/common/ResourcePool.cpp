//===- ResourcePool.cpp - A pool of identical resources -------------------===//

#include "common/ResourcePool.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

using namespace reef_perf;

ResourcePool::ResourcePool(std::string Name, uint32_t Count)
    : Name(std::move(Name)), FreeAt(Count, 0) {
  if (Count == 0)
    throw std::invalid_argument("pool '" + this->Name +
                                "' needs at least one resource");
}

Cycle ResourcePool::getNextFree(Cycle Earliest) const {
  return std::max(Earliest, *std::ranges::min_element(FreeAt));
}

Cycle ResourcePool::reserve(Cycle Earliest, uint64_t Occupancy) {
  return book(*std::ranges::min_element(FreeAt), Earliest, Occupancy);
}

uint32_t ResourcePool::getEarliestFree() const {
  return static_cast<uint32_t>(std::ranges::min_element(FreeAt) -
                               FreeAt.begin());
}

Cycle ResourcePool::reserveOn(uint32_t Index, Cycle Earliest,
                              uint64_t Occupancy) {
  return book(FreeAt.at(Index), Earliest, Occupancy);
}

Cycle ResourcePool::reserveFirst(Cycle Earliest, uint64_t Occupancy) {
  return book(FreeAt.front(), Earliest, Occupancy);
}

Cycle ResourcePool::book(Cycle &FreeAtCycle, Cycle Earliest,
                         uint64_t Occupancy) {
  Cycle Start = std::max(Earliest, FreeAtCycle);
  FreeAtCycle = Start + Occupancy;
  ++Ops;
  BusyCycles += Occupancy;
  return Start;
}
