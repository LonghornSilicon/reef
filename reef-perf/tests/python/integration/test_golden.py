"""Golden timing check: refactors must not change what the model computes.

Every microbenchmark runs on the M3 config and its headline numbers are
compared, exactly, with tests/golden/m3_results.json. A change that moves
code around must leave them identical; a change that is meant to alter timing
updates the golden file in the same commit and says why.

To (re)generate the golden file from the current code:

    REEF_PERF_UPDATE_GOLDEN=1 uv run pytest \
        tests/python/integration/test_golden.py

A second golden file, tests/golden/pinned_results.json, pins the same
numbers on the pinned-RTL config (m3.yaml + configs/machines/coral_pinned.yaml)
for the core, frontend and vector-unit microbenchmarks. Work on that config
regenerates it with its own variable, so the M3 file cannot move with it:

    REEF_PERF_UPDATE_PINNED_GOLDEN=1 uv run pytest \
        tests/python/integration/test_golden.py

Only keys present in the golden file are compared, so adding a new counter
(for example a new stall reason) does not break the check. Pool utilization
is not compared: it describes how work is spread over resources, which modules
are free to reorganize.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from reef_perf.experiments.correlation import pinned
from reef_perf.sim import MODULE_ROOT, SimResult, run_sim

GOLDEN_PATH = MODULE_ROOT / "tests" / "golden" / "m3_results.json"
"""Expected results, one entry per microbenchmark."""

PINNED_GOLDEN_PATH = MODULE_ROOT / "tests" / "golden" / "pinned_results.json"
"""Expected results on the pinned-RTL config, one entry per "suite/name"."""

UPDATE_ENV = "REEF_PERF_UPDATE_GOLDEN"
"""Set to 1 to rewrite the M3 golden file instead of checking it."""

UPDATE_PINNED_ENV = "REEF_PERF_UPDATE_PINNED_GOLDEN"
"""Set to 1 to rewrite the pinned golden file instead of checking it."""


def summarize(result: SimResult) -> dict[str, Any]:
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
        expected: One microbenchmark's golden entry.
        actual: The same microbenchmark's summary now.

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


def check_golden(
    path: Path, actual: dict[str, dict[str, Any]], update_env: str
) -> None:
    """Compare results with a golden file, or rewrite it on request.

    Args:
        path: The golden file.
        actual: Summary by program name.
        update_env: Environment variable that, set to 1, rewrites the file.
    """
    if os.environ.get(update_env) == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, indent=2) + "\n")
        pytest.skip(f"wrote {path}")

    if not path.is_file():
        pytest.fail(
            f"{path} is missing; generate it with {update_env}=1",
            pytrace=False,
        )
    golden = json.loads(path.read_text())
    names = sorted(actual)
    assert sorted(golden) == names, "microbenchmark list changed"
    problems = {
        name: diff
        for name in names
        if (diff := mismatches(golden[name], actual[name]))
    }
    assert not problems, json.dumps(problems, indent=2)


@pytest.mark.integration
@pytest.mark.slow
def test_timing_matches_golden(
    cpp_build_dir: Path, microbenchmark_elfs: dict[str, Path]
) -> None:
    """Every microbenchmark's cycles and counters match the golden file."""
    del cpp_build_dir  # built for its side effect
    names = sorted(microbenchmark_elfs)
    with ThreadPoolExecutor() as pool:
        results = list(
            pool.map(lambda n: run_sim(microbenchmark_elfs[n]), names)
        )
    actual = {
        name: summarize(result)
        for name, result in zip(names, results, strict=True)
    }

    check_golden(GOLDEN_PATH, actual, UPDATE_ENV)


@pytest.mark.integration
@pytest.mark.slow
def test_pinned_timing_matches_golden(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """Every microbenchmark on the pinned-RTL config matches its golden."""
    del cpp_build_dir  # built for its side effect
    elfs = {
        f"{suite}/{name}": elf
        for (suite, name), elf in pinned.build_programs(tmp_path).items()
        if suite != "kernels"
    }
    names = sorted(elfs)
    with ThreadPoolExecutor() as pool:
        results = list(
            pool.map(
                lambda n: run_sim(
                    elfs[n],
                    overlay_configs=[pinned.PINNED_CONFIG],
                    binary=pinned.find_pinned_binary(),
                ),
                names,
            )
        )
    actual = {
        name: summarize(result)
        for name, result in zip(names, results, strict=True)
    }
    check_golden(PINNED_GOLDEN_PATH, actual, UPDATE_PINNED_ENV)
