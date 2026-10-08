#pragma once

/** @file
 *  @brief Base class of the model's top-level modules (frontend, backend,
 *         vector, matrix, memory).
 *
 *  A module owns a subtree of the Sparta tree, top.MODULE, and the units
 *  under it, top.MODULE.UNIT. It wires its own units together; the
 *  simulation only binds the *public* ports each module lists in its header,
 *  so a module can add, remove or rename internal units without touching any
 *  other module. See docs/interfaces.md.
 */

#include "reef_perf/common/resource_pool.hpp"

#include "sparta/ports/Port.hpp"
#include "sparta/simulation/ResourceFactory.hpp"
#include "sparta/simulation/ResourceTreeNode.hpp"
#include "sparta/simulation/TreeNode.hpp"

#include <memory>
#include <string>
#include <vector>

namespace reef_perf {

/// Tree nodes created by modules; the simulation deletes them at teardown.
using NodeList = std::vector<std::unique_ptr<sparta::TreeNode>>;

/** One top-level module of the model.
 *
 *  Life cycle, driven by ReefSim:
 *  1. add_factories(): register a Sparta ResourceFactory per unit.
 *  2. build(): create top.MODULE and one tree node per unit.
 *  3. (Sparta creates the units and applies the configuration.)
 *  4. bind(): bind the ports between this module's own units and look up
 *     the unit objects.
 *  5. ReefSim binds the public ports between modules.
 */
class Module {
  public:
    /** Names the module.
     *
     *  @param name Tree node name, e.g. "backend".
     *  @param description Tree node description.
     */
    Module(std::string name, std::string description);

    /// Modules hold only non-owning pointers into the tree.
    virtual ~Module() = default;

    /// Not copyable: units keep pointers to their module's objects.
    Module(const Module&) = delete;
    /** Not copy-assignable.
     *  @return Never returns.
     */
    Module& operator=(const Module&) = delete;
    /// Not movable, for the same reason.
    Module(Module&&) = delete;
    /** Not move-assignable.
     *  @return Never returns.
     */
    Module& operator=(Module&&) = delete;

    /** Name of the module's tree node.
     *
     *  @return The name, e.g. "backend".
     */
    [[nodiscard]] const std::string& name() const { return name_; }

    /** Registers a ResourceFactory for each of the module's units.
     *
     *  Unit names must be unique across all modules, because the simulation
     *  keeps one resource set.
     *
     *  @param resources The simulation's resource set.
     */
    virtual void add_factories(sparta::ResourceSet& resources) = 0;

    /** Creates top.MODULE and a tree node for every unit.
     *
     *  @param parent The tree root ("top").
     *  @param resources The resource set the factories were added to.
     *  @param nodes Receives the new nodes, which the simulation deletes.
     */
    void build(sparta::TreeNode* parent, sparta::ResourceSet& resources,
               NodeList& nodes);

    /** Binds the ports between this module's own units and looks up the
     *  unit objects. Called after Sparta has created the units.
     */
    virtual void bind() = 0;

    /** Resource pools to report at the end of the run.
     *
     *  @return The module's pools, in reporting order.
     */
    [[nodiscard]] virtual std::vector<const ResourcePool*> pools() const {
        return {};
    }

    /** Path of one of the module's ports, relative to the tree root.
     *
     *  @param path Path relative to the module, e.g.
     *              "dispatch.ports.in_insts".
     *  @return The path relative to "top", e.g.
     *          "backend.dispatch.ports.in_insts".
     */
    [[nodiscard]] std::string port_path(const std::string& path) const {
        return name_ + "." + path;
    }

  protected:
    /** Names of the units build() creates, in order. Each must have a
     *  factory registered by add_factories().
     *
     *  @return Unit names.
     */
    [[nodiscard]] virtual std::vector<std::string> unit_names() const = 0;

    /** One of the module's units. Valid in and after bind().
     *
     *  @tparam UnitT The unit's class.
     *  @param unit_name Unit name.
     *  @return The unit.
     */
    template <class UnitT> UnitT* unit(const std::string& unit_name) const {
        return node_->getChildAs<sparta::ResourceTreeNode>(unit_name)
            ->getResourceAs<UnitT>();
    }

    /** One of the module's ports. Valid in and after bind().
     *
     *  @param path Path relative to the module, e.g. "rob.ports.out_credits".
     *  @return The port.
     */
    [[nodiscard]] sparta::Port* port(const std::string& path) const {
        return node_->getChildAs<sparta::Port>(path);
    }

  private:
    /// Tree node name.
    std::string name_;
    /// Tree node description.
    std::string description_;
    /// top.MODULE, set by build().
    sparta::TreeNode* node_ = nullptr;
};

} // namespace reef_perf
