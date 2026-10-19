//===- Core.cpp - The core module (scalar pipeline) -----------------------===//

#include "core/Core.h"
#include "core/Lsu.h"
#include "core/Rob.h"
#include "core/ScalarExec.h"

#include "sparta/ports/Port.hpp"
#include "sparta/simulation/ResourceFactory.hpp"

#include <cstdint>
#include <string>
#include <vector>

using namespace reef_perf;

Core::Core()
    : Module("core", "Dispatch, scalar execution, the LSU and retirement") {}

void Core::addFactories(sparta::ResourceSet &Resources) {
  Resources.addResourceFactory<
      sparta::ResourceFactory<Dispatch, Dispatch::DispatchParameterSet>>();
  Resources.addResourceFactory<sparta::ResourceFactory<
      ScalarExec, ScalarExec::ScalarExecParameterSet>>();
  Resources
      .addResourceFactory<sparta::ResourceFactory<Lsu, Lsu::LsuParameterSet>>();
  Resources
      .addResourceFactory<sparta::ResourceFactory<Rob, Rob::RobParameterSet>>();
}

std::vector<std::string> Core::getUnitNames() const {
  return {Dispatch::name, ScalarExec::name, Lsu::name, Rob::name};
}

void Core::bind() {
  sparta::bind(getChildPort("dispatch.ports.out_scalar"),
               getChildPort("scalar_exec.ports.in_insts"));
  sparta::bind(getChildPort("dispatch.ports.out_lsu"),
               getChildPort("lsu.ports.in_insts"));
  sparta::bind(getChildPort("lsu.ports.out_credits"),
               getChildPort("dispatch.ports.in_lsu_credits"));
  sparta::bind(getChildPort("dispatch.ports.out_rob"),
               getChildPort("rob.ports.in_insts"));
  sparta::bind(getChildPort("rob.ports.out_credits"),
               getChildPort("dispatch.ports.in_rob_credits"));

  TheDispatch = getChildUnit<Dispatch>(Dispatch::name);
  TheScalarExec = getChildUnit<ScalarExec>(ScalarExec::name);
  TheLsu = getChildUnit<Lsu>(Lsu::name);
  TheRob = getChildUnit<Rob>(Rob::name);
}

void Core::setMemory(MemoryInterface *Mem) { TheLsu->setMemory(Mem); }

void Core::setInstBuffer(InstBufferInterface *Buffer) {
  TheDispatch->setInstBuffer(Buffer);
}

std::vector<const ResourcePool *> Core::getPools() const {
  std::vector<const ResourcePool *> Pools = TheScalarExec->getPools();
  for (const ResourcePool *Pool : TheLsu->getPools())
    Pools.push_back(Pool);
  return Pools;
}

uint64_t Core::getNumRetired() const { return TheRob->getNumRetired(); }

uint64_t Core::getLastRetireCycle() const {
  return TheRob->getLastRetireCycle();
}

uint64_t Core::getCyclesWithDispatch() const {
  return TheDispatch->getCyclesWithDispatch();
}

uint64_t Core::getStallCycles(StallReason Reason) const {
  return TheDispatch->getStallCycles(Reason);
}

const std::vector<uint64_t> &Core::getMcycleReads() const {
  return TheRob->getMcycleReads();
}
