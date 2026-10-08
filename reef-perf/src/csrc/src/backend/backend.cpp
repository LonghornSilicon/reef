#include "reef_perf/backend/backend.hpp"

#include "reef_perf/backend/lsu.hpp"
#include "reef_perf/backend/rob.hpp"
#include "reef_perf/backend/scalar_exec.hpp"

#include "sparta/ports/Port.hpp"
#include "sparta/simulation/ResourceFactory.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace reef_perf {

Backend::Backend()
    : Module("backend", "Dispatch, scalar execution and retirement") {}

void Backend::add_factories(sparta::ResourceSet& resources) {
    resources.addResourceFactory<
        sparta::ResourceFactory<Dispatch, Dispatch::DispatchParameterSet>>();
    resources.addResourceFactory<sparta::ResourceFactory<
        ScalarExec, ScalarExec::ScalarExecParameterSet>>();
    resources.addResourceFactory<
        sparta::ResourceFactory<Lsu, Lsu::LsuParameterSet>>();
    resources.addResourceFactory<
        sparta::ResourceFactory<Rob, Rob::RobParameterSet>>();
}

std::vector<std::string> Backend::unit_names() const {
    return {Dispatch::name, ScalarExec::name, Lsu::name, Rob::name};
}

void Backend::bind() {
    sparta::bind(port("dispatch.ports.out_scalar"),
                 port("scalar_exec.ports.in_insts"));
    sparta::bind(port("dispatch.ports.out_lsu"), port("lsu.ports.in_insts"));
    sparta::bind(port("lsu.ports.out_credits"),
                 port("dispatch.ports.in_lsu_credits"));
    sparta::bind(port("dispatch.ports.out_rob"), port("rob.ports.in_insts"));
    sparta::bind(port("rob.ports.out_credits"),
                 port("dispatch.ports.in_rob_credits"));

    dispatch_ = unit<Dispatch>(Dispatch::name);
    scalar_exec_ = unit<ScalarExec>(ScalarExec::name);
    lsu_ = unit<Lsu>(Lsu::name);
    rob_ = unit<Rob>(Rob::name);
}

std::vector<const ResourcePool*> Backend::pools() const {
    std::vector<const ResourcePool*> all = scalar_exec_->pools();
    for (const ResourcePool* pool : lsu_->pools()) {
        all.push_back(pool);
    }
    return all;
}

std::uint64_t Backend::num_retired() const { return rob_->num_retired(); }

std::uint64_t Backend::last_retire_cycle() const {
    return rob_->last_retire_cycle();
}

std::uint64_t Backend::cycles_with_dispatch() const {
    return dispatch_->cycles_with_dispatch();
}

std::uint64_t Backend::stall_cycles(StallReason reason) const {
    return dispatch_->stall_cycles(reason);
}

} // namespace reef_perf
