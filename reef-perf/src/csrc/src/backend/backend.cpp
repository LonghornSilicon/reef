#include "reef_perf/backend/backend.hpp"

#include "reef_perf/backend/execute.hpp"
#include "reef_perf/backend/rob.hpp"

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
    resources.addResourceFactory<
        sparta::ResourceFactory<Execute, Execute::ExecuteParameterSet>>();
    resources.addResourceFactory<
        sparta::ResourceFactory<Rob, Rob::RobParameterSet>>();
}

std::vector<std::string> Backend::unit_names() const {
    return {Dispatch::name, Execute::name, Rob::name};
}

void Backend::bind() {
    sparta::bind(port("dispatch.ports.out_execute"),
                 port("execute.ports.in_insts"));
    sparta::bind(port("dispatch.ports.out_rob"), port("rob.ports.in_insts"));
    sparta::bind(port("rob.ports.out_credits"),
                 port("dispatch.ports.in_rob_credits"));
    sparta::bind(port("execute.ports.out_lsu_credits"),
                 port("dispatch.ports.in_lsu_credits"));
    sparta::bind(port("execute.ports.out_vec_credits"),
                 port("dispatch.ports.in_vec_credits"));

    dispatch_ = unit<Dispatch>(Dispatch::name);
    execute_ = unit<Execute>(Execute::name);
    rob_ = unit<Rob>(Rob::name);
}

std::vector<const ResourcePool*> Backend::pools() const {
    return execute_->pools();
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
