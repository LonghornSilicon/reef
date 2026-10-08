#include "reef_perf/matrix/matrix.hpp"

#include "reef_perf/matrix/mxu.hpp"

#include "sparta/simulation/ResourceFactory.hpp"

#include <string>
#include <vector>

namespace reef_perf {

Matrix::Matrix() : Module("matrix", "Matrix engine") {}

void Matrix::add_factories(sparta::ResourceSet& resources) {
    resources.addResourceFactory<
        sparta::ResourceFactory<Mxu, Mxu::MxuParameterSet>>();
}

std::vector<std::string> Matrix::unit_names() const { return {Mxu::name}; }

void Matrix::bind() { mxu_ = unit<Mxu>(Mxu::name); }

std::vector<const ResourcePool*> Matrix::pools() const {
    return mxu_->pools();
}

} // namespace reef_perf
