#include "reef_perf/common/module.hpp"

#include <memory>
#include <string>
#include <utility>

namespace reef_perf {

Module::Module(std::string name, std::string description)
    : name_(std::move(name)), description_(std::move(description)) {}

void Module::build(sparta::TreeNode* parent, sparta::ResourceSet& resources,
                   NodeList& nodes) {
    auto module_node =
        std::make_unique<sparta::TreeNode>(parent, name_, description_);
    node_ = module_node.get();
    for (const std::string& unit : unit_names()) {
        nodes.emplace_back(std::make_unique<sparta::ResourceTreeNode>(
            node_, unit, sparta::TreeNode::GROUP_NAME_NONE,
            sparta::TreeNode::GROUP_IDX_NONE, unit,
            resources.getResourceFactory(unit)));
    }
    // Children first, as before: the simulation deletes nodes in this order.
    nodes.emplace_back(std::move(module_node));
}

} // namespace reef_perf
