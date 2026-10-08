#include "reef_perf/vector/vector.hpp"

#include "reef_perf/vector/vxu.hpp"

#include "sparta/simulation/ResourceFactory.hpp"

#include <string>
#include <vector>

namespace reef_perf {

Vector::Vector() : Module("vector", "RVV backend") {}

void Vector::add_factories(sparta::ResourceSet& resources) {
    resources.addResourceFactory<
        sparta::ResourceFactory<Vxu, Vxu::VxuParameterSet>>();
}

std::vector<std::string> Vector::unit_names() const { return {Vxu::name}; }

void Vector::bind() { vxu_ = unit<Vxu>(Vxu::name); }

std::vector<const ResourcePool*> Vector::pools() const { return vxu_->pools(); }

} // namespace reef_perf
