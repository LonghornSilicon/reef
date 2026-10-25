"""Top-down analysis: where the dispatch slots went.

Every cycle the core has `dispatch_width` dispatch slots. The simulator
charges each slot to exactly one category (docs/topdown.md); this module turns
those counts into a tree and renders it:

    slots
      retiring                instructions dispatched
      frontend_bound          nothing to dispatch
        branch_resteer
        fetch_bandwidth
      backend_bound           an instruction was blocked
        core | memory | vector | matrix     the unit it waited for
          raw_hazard | waw_hazard | rob_full | <unit>_queue_full
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Node:
    """One category of dispatch slots.

    Attributes:
        name: Category name, e.g. "backend_bound" or "raw_hazard".
        slots: Slots in this category, including its children.
        children: Subcategories; their slots add up to this node's.
    """

    name: str
    slots: int
    children: tuple["Node", ...] = field(default=())

    def find(self, path: str) -> "Node":
        """Look up a descendant by a dotted path.

        Args:
            path: E.g. "backend_bound.memory.raw_hazard".

        Returns:
            The node.

        Raises:
            KeyError: If no such node exists.
        """
        node = self
        for name in path.split("."):
            matches = [child for child in node.children if child.name == name]
            if not matches:
                raise KeyError(path)
            node = matches[0]
        return node

    def level(self, depth: int) -> dict[str, int]:
        """Flatten the tree at one depth.

        Leaves above that depth are kept as they are, so the result always
        covers every slot.

        Args:
            depth: 1 for retiring/frontend_bound/backend_bound, 2 for their
                children, and so on.

        Returns:
            Dotted path -> slots, for every node at that depth.
        """
        found: dict[str, int] = {}

        def visit(node: Node, prefix: str, remaining: int) -> None:
            for child in node.children:
                path = f"{prefix}{child.name}"
                if remaining == 1 or not child.children:
                    found[path] = child.slots
                else:
                    visit(child, f"{path}.", remaining - 1)

        visit(self, "", depth)
        return found


def _group(name: str, counts: Mapping[str, int]) -> Node:
    """Build a node with one child per entry of counts."""
    children = tuple(Node(key, int(value)) for key, value in counts.items())
    return Node(name, sum(child.slots for child in children), children)


def build_tree(topdown: Mapping[str, Any]) -> Node:
    """Build the top-down tree from the simulator's "topdown" JSON object.

    Args:
        topdown: SimResult.topdown.

    Returns:
        The root, named "slots".

    Raises:
        ValueError: If the categories don't add up to the slot total.
    """
    frontend = _group("frontend_bound", topdown["frontend_bound"])
    backend_children = tuple(
        _group(bound, reasons)
        for bound, reasons in topdown["backend_bound"].items()
    )
    backend = Node(
        "backend_bound",
        sum(child.slots for child in backend_children),
        backend_children,
    )
    retiring = Node("retiring", int(topdown["retiring"]))
    root = Node("slots", int(topdown["slots"]), (retiring, frontend, backend))
    counted = sum(child.slots for child in root.children)
    if counted != root.slots:
        raise ValueError(
            f"top-down categories add up to {counted}, not {root.slots}"
        )
    return root


def render(root: Node, hide_zero: bool = True) -> str:
    """Render the tree as indented text with percentages of all slots.

    Args:
        root: Output of build_tree().
        hide_zero: Leave out categories with no slots.

    Returns:
        One line per category.
    """
    lines = []

    def visit(node: Node, depth: int) -> None:
        for child in node.children:
            if hide_zero and child.slots == 0:
                continue
            share = 100 * child.slots / root.slots if root.slots else 0.0
            label = "  " * depth + child.name
            lines.append(f"{label:<28} {child.slots:>10} {share:6.1f}%")
            visit(child, depth + 1)

    visit(root, 0)
    return "\n".join(lines)
