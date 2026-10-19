//===- ResourcePool.cpp - A pool of identical units -----------------------===//

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
                                "' needs at least one unit");
}

Cycle ResourcePool::getNextFree(Cycle Earliest) const {
  return std::max(Earliest, *std::ranges::min_element(FreeAt));
}

Cycle ResourcePool::reserve(Cycle Earliest, uint64_t Occupancy) {
  auto Unit = std::ranges::min_element(FreeAt);
  Cycle Start = std::max(Earliest, *Unit);
  *Unit = Start + Occupancy;
  ++Ops;
  BusyCycles += Occupancy;
  return Start;
}
