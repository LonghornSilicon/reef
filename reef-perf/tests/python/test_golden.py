"""Golden timing check: refactors must not change what the model computes.

Every workload runs on the M3 config and its headline numbers are compared,
exactly, with tests/python/golden/m3.json. A change that moves code around
must leave them identical; a change that is meant to alter timing updates the
golden file in the same commit and says why.

To (re)generate the golden file from the current code:

    REEF_PERF_UPDATE_GOLDEN=1 uv run pytest tests/python/test_golden.py

Only keys present in the golden file are compared, so adding a new counter
(for example a new stall reason) does not break the check. Pool utilisation
is not compared: it describes how work is spread over units, which modules are
free to reorganise.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from reef_perf.sim import SimResult, run_sim

GOLDEN_PATH = Path(__file__).parent / "golden" / "m3.json"
"""Expected results, one entry per workload."""

UPDATE_ENV = "REEF_PERF_UPDATE_GOLDEN"
"""Set to 1 to rewrite the golden file instead of checking it."""


def summarise(result: SimResult) -> dict[str, Any]:
    """Pick the numbers a refactor must preserve.

    Args:
        result: One simulation result.

    Returns:
        Instruction counts, cycles, redirects and dispatch counters.
    """
    return {
        "instructions": result.instructions,
        "functional_instructions": result.functional_instructions,
        "cycles": result.cycles,
        "taken_redirects": result.taken_redirects,
        "dispatch": dict(sorted(result.dispatch.items())),
    }


def mismatches(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """List the golden values that differ, ignoring keys new since then.

    Args:
        expected: One workload's golden entry.
        actual: The same workload's summary now.

    Returns:
        One "key: expected -> actual" line per difference.
    """
    found = []
    for key, want in expected.items():
        if isinstance(want, dict):
            have = actual.get(key, {})
            found += [
                f"{key}.{sub}: {value} -> {have.get(sub)}"
                for sub, value in want.items()
                if have.get(sub) != value
            ]
        elif actual.get(key) != want:
            found.append(f"{key}: {want} -> {actual.get(key)}")
    return found


@pytest.mark.integration
@pytest.mark.slow
def test_timing_matches_golden(
    cpp_build_dir: Path, workload_elfs: dict[str, Path]
) -> None:
    """Every workload's cycles and counters match the golden file."""
    del cpp_build_dir  # built for its side effect
    names = sorted(workload_elfs)
    with ThreadPoolExecutor() as pool:
        results = list(pool.map(lambda n: run_sim(workload_elfs[n]), names))
    actual = {
        name: summarise(result)
        for name, result in zip(names, results, strict=True)
    }

    if os.environ.get(UPDATE_ENV) == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(actual, indent=2) + "\n")
        pytest.skip(f"wrote {GOLDEN_PATH}")

    if not GOLDEN_PATH.is_file():
        pytest.fail(
            f"{GOLDEN_PATH} is missing; generate it with {UPDATE_ENV}=1",
            pytrace=False,
        )
    golden = json.loads(GOLDEN_PATH.read_text())
    assert sorted(golden) == names, "workload list changed"
    problems = {
        name: diff
        for name in names
        if (diff := mismatches(golden[name], actual[name]))
    }
    assert not problems, json.dumps(problems, indent=2)
