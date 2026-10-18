//===- Backend.cpp - The backend module -----------------------------------===//

#include "backend/Backend.h"
#include "backend/Lsu.h"
#include "backend/Rob.h"
#include "backend/ScalarExec.h"

#include "sparta/ports/Port.hpp"
#include "sparta/simulation/ResourceFactory.hpp"

#include <cstdint>
#include <string>
#include <vector>

using namespace reef_perf;

Backend::Backend()
    : Module("backend", "Dispatch, scalar execution and retirement") {}

void Backend::addFactories(sparta::ResourceSet &Resources) {
  Resources.addResourceFactory<
      sparta::ResourceFactory<Dispatch, Dispatch::DispatchParameterSet>>();
  Resources.addResourceFactory<sparta::ResourceFactory<
      ScalarExec, ScalarExec::ScalarExecParameterSet>>();
  Resources
      .addResourceFactory<sparta::ResourceFactory<Lsu, Lsu::LsuParameterSet>>();
  Resources
      .addResourceFactory<sparta::ResourceFactory<Rob, Rob::RobParameterSet>>();
}

std::vector<std::string> Backend::getUnitNames() const {
  return {Dispatch::name, ScalarExec::name, Lsu::name, Rob::name};
}

void Backend::bind() {
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

void Backend::setMemory(MemoryInterface *Mem) { TheLsu->setMemory(Mem); }

void Backend::setInstBuffer(InstBufferInterface *Buffer) {
  TheDispatch->setInstBuffer(Buffer);
}

std::vector<const ResourcePool *> Backend::getPools() const {
  std::vector<const ResourcePool *> Pools = TheScalarExec->getPools();
  for (const ResourcePool *Pool : TheLsu->getPools())
    Pools.push_back(Pool);
  return Pools;
}

uint64_t Backend::getNumRetired() const { return TheRob->getNumRetired(); }

uint64_t Backend::getLastRetireCycle() const {
  return TheRob->getLastRetireCycle();
}

uint64_t Backend::getCyclesWithDispatch() const {
  return TheDispatch->getCyclesWithDispatch();
}

uint64_t Backend::getStallCycles(StallReason Reason) const {
  return TheDispatch->getStallCycles(Reason);
}
