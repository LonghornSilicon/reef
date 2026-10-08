#include "reef_perf/frontend/frontend.hpp"

#include "sparta/simulation/ResourceFactory.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace reef_perf {

Frontend::Frontend() : Module("frontend", "Fetch, Spike and decode") {}

void Frontend::add_factories(sparta::ResourceSet& resources) {
    resources.addResourceFactory<
        sparta::ResourceFactory<Fetch, Fetch::FetchParameterSet>>();
}

std::vector<std::string> Frontend::unit_names() const { return {Fetch::name}; }

void Frontend::bind() { fetch_ = unit<Fetch>(Fetch::name); }

void Frontend::set_func_sim(FuncSim* funcsim) { fetch_->set_func_sim(funcsim); }

std::uint64_t Frontend::num_redirects() const {
    return fetch_->num_redirects();
}

} // namespace reef_perf
