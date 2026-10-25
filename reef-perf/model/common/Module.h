//===- Module.h - Top-level modules -----------------------------*- C++ -*-===//
///
/// \file
/// The model's top-level modules (frontend, core, vector, matrix, memory).
/// A module is one of the five owner-assignable parts of the model; a unit
/// is a Sparta unit, and a module has one or more (docs/glossary.md).
/// A module owns the Sparta subtree top.MODULE and wires its own units; the
/// simulation binds only the public ports each module lists, so a module can
/// change its insides without touching the others. See docs/interfaces.md.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_COMMON_MODULE_H
#define REEF_PERF_COMMON_MODULE_H

#include "common/ResourcePool.h"

#include "sparta/ports/Port.hpp"
#include "sparta/simulation/ResourceFactory.hpp"
#include "sparta/simulation/ResourceTreeNode.hpp"
#include "sparta/simulation/TreeNode.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace reef_perf {

/// Tree nodes created by modules; the simulation deletes them at teardown.
using NodeList = std::vector<std::unique_ptr<sparta::TreeNode>>;

/// One top-level module. ReefSim calls addFactories(), then build(); Sparta
/// creates and configures the units; then ReefSim calls bind() and binds the
/// public ports between modules.
class Module {
public:
  /// Creates a module whose tree node is top.\p Name.
  Module(std::string Name, std::string Description)
      : Name(std::move(Name)), Description(std::move(Description)) {}
  virtual ~Module() = default; ///< Holds no tree state.

  /// \returns the tree node name, e.g. "core".
  const std::string &getName() const { return Name; }

  /// \returns the tree node description.
  const std::string &getDescription() const { return Description; }

  /// Registers a ResourceFactory per unit in \p Resources. Unit names must be
  /// unique across modules (the simulation has one resource set).
  virtual void addFactories(sparta::ResourceSet &Resources) = 0;

  /// Creates top.MODULE under \p Parent with one child node per
  /// getUnitNames() entry, using the factories in \p Resources. The new nodes
  /// are appended to \p Nodes.
  virtual void build(sparta::TreeNode *Parent, sparta::ResourceSet &Resources,
                     NodeList &Nodes);

  /// Binds the ports between the module's own units, after Sparta has
  /// created them.
  virtual void bind() = 0;

  /// \returns the module's resource pools, in reporting order.
  virtual std::vector<const ResourcePool *> getPools() const { return {}; }

  /// \returns \p Path, a port path relative to the module, relative to the
  /// tree root instead.
  std::string getPortPath(const std::string &Path) const {
    return Name + "." + Path;
  }

protected:
  /// \returns the units build() creates under top.MODULE.
  virtual std::vector<std::string> getUnitNames() const { return {}; }

  /// \returns the unit \p UnitName created by build(); valid from bind() on.
  template <class UnitT>
  UnitT *getChildUnit(const std::string &UnitName) const {
    return Node->getChildAs<sparta::ResourceTreeNode>(UnitName)
        ->getResourceAs<UnitT>();
  }

  /// \returns the port at \p Path, relative to the module; valid from bind()
  /// on.
  sparta::Port *getChildPort(const std::string &Path) const {
    return Node->getChildAs<sparta::Port>(Path);
  }

private:
  std::string Name;
  std::string Description;
  sparta::TreeNode *Node = nullptr; // top.MODULE, set by build().
};

/// A module that is a single unit: top.MODULE is the unit itself, so its
/// parameters are top.MODULE.params.NAME and its ports top.MODULE.ports.NAME.
template <class UnitT, class ParamsT> class UnitModule final : public Module {
public:
  using Module::Module;

  /// Registers the unit's factory in \p Resources.
  void addFactories(sparta::ResourceSet &Resources) override {
    Resources.addResourceFactory<sparta::ResourceFactory<UnitT, ParamsT>>();
  }

  /// Creates top.MODULE under \p Parent as the unit's own node, using the
  /// factory in \p Resources, and appends it to \p Nodes.
  void build(sparta::TreeNode *Parent, sparta::ResourceSet &Resources,
             NodeList &Nodes) override {
    auto NewNode = std::make_unique<sparta::ResourceTreeNode>(
        Parent, getName(), sparta::TreeNode::GROUP_NAME_NONE,
        sparta::TreeNode::GROUP_IDX_NONE, getDescription(),
        Resources.getResourceFactory(UnitT::name));
    UnitNode = NewNode.get();
    Nodes.emplace_back(std::move(NewNode));
  }

  /// Looks up the unit.
  void bind() override { TheUnit = UnitNode->getResourceAs<UnitT>(); }

  /// \returns the unit's pools, if it has any.
  std::vector<const ResourcePool *> getPools() const override {
    if constexpr (requires(const UnitT &U) { U.getPools(); })
      return TheUnit->getPools();
    else
      return {};
  }

  /// \returns the unit; valid from bind() on.
  UnitT *getUnit() const { return TheUnit; }

private:
  sparta::ResourceTreeNode *UnitNode = nullptr;
  UnitT *TheUnit = nullptr;
};

} // namespace reef_perf

#endif // REEF_PERF_COMMON_MODULE_H
