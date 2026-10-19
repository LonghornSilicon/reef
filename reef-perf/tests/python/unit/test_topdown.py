"""Check the top-down tree built from the simulator's slot counts."""

import pytest

from reef_perf.topdown import build_tree, render

SAMPLE = {
    "dispatch_width": 4,
    "slots": 100,
    "retiring": 40,
    "frontend_bound": {"branch_resteer": 10, "fetch_bandwidth": 5},
    "backend_bound": {
        "core": {"raw_hazard": 20},
        "memory": {"raw_hazard": 15, "lsu_queue_full": 10},
        "vector": {},
        "matrix": {},
    },
}


@pytest.mark.unit
def test_tree_adds_up() -> None:
    """Every node holds the sum of its children."""
    tree = build_tree(SAMPLE)
    assert tree.find("retiring").slots == 40
    assert tree.find("frontend_bound").slots == 15
    assert tree.find("backend_bound").slots == 45
    assert tree.find("backend_bound.memory").slots == 25
    assert tree.find("backend_bound.memory.lsu_queue_full").slots == 10


@pytest.mark.unit
def test_levels_cover_every_slot() -> None:
    """Flattening at any depth keeps all slots, including shallow leaves."""
    tree = build_tree(SAMPLE)
    assert tree.level(1) == {
        "retiring": 40,
        "frontend_bound": 15,
        "backend_bound": 45,
    }
    level2 = tree.level(2)
    assert level2["retiring"] == 40
    assert level2["backend_bound.vector"] == 0
    for depth in (1, 2, 3):
        assert sum(tree.level(depth).values()) == 100


@pytest.mark.unit
def test_rejects_counts_that_do_not_add_up() -> None:
    """A slot total that the categories don't explain is an error."""
    with pytest.raises(ValueError, match="add up to 100, not 101"):
        build_tree(SAMPLE | {"slots": 101})


@pytest.mark.unit
def test_render_shows_shares_and_hides_zeros() -> None:
    """The text tree is indented, with each category's share of all slots."""
    text = render(build_tree(SAMPLE))
    assert "retiring" in text and "40.0%" in text
    assert "    lsu_queue_full" in text
    assert "vector" not in text
