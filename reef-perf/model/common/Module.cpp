//===- Module.cpp - Top-level modules -------------------------------------===//

#include "common/Module.h"

#include <memory>
#include <string>
#include <utility>

using namespace reef_perf;

void Module::build(sparta::TreeNode *Parent, sparta::ResourceSet &Resources,
                   NodeList &Nodes) {
  auto ModuleNode =
      std::make_unique<sparta::TreeNode>(Parent, Name, Description);
  Node = ModuleNode.get();
  for (const std::string &Unit : getUnitNames())
    Nodes.emplace_back(std::make_unique<sparta::ResourceTreeNode>(
        Node, Unit, sparta::TreeNode::GROUP_NAME_NONE,
        sparta::TreeNode::GROUP_IDX_NONE, Unit,
        Resources.getResourceFactory(Unit)));
  // Children first: the simulation deletes nodes in this order.
  Nodes.emplace_back(std::move(ModuleNode));
}
