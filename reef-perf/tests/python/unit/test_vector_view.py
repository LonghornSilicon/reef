"""Check lane placement in the vector view."""

from pathlib import Path

import pytest

from reef_perf.vector_view import (
    VectorParams,
    load_params,
    parse_vector_run,
    render_html,
)

PARAMS = VectorParams(queue_entries=8, lanes=2, latency=4, to_scalar_latency=2)

# (seq, queued, issue, result_ready, disassembly). Two 4-cycle adds fill both
# lanes at cycle 10, so the third waits until lane 0 frees at 14.
ROWS = [
    (1, 9, 9, 9, "vsetvli t0, a3, e32, m4, ta, ma"),
    (2, 10, 10, 16, "vadd.vv v4, v0, v0"),
    (3, 10, 10, 16, "vmul.vv v8, v0, v0"),
    (4, 11, 14, 20, "vadd.vv v12, v0, v0"),
]


def write_trace(path: Path, rows: list[tuple[int, int, int, int, str]]) -> None:
    """Writes rows in the `reef_perf --trace` format."""
    lines = ["cycle\tactivity\tseq\tpc\tinstruction"]
    for seq, queued, issue, ready, disasm in rows:
        tail = f"{seq}\t0x{seq:x}\t{disasm}"
        lines += [
            f"{queued}\tVector_Buffer\t{tail}",
            f"{issue}\tVector\t{tail}",
            f"{ready}\tresult_ready\t{tail}",
        ]
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.unit
def test_places_instructions_on_lanes(tmp_path: Path) -> None:
    """Lane and occupancy follow ResourcePool::reserve and the latency model."""
    trace = tmp_path / "t.tsv"
    write_trace(trace, ROWS)
    run = parse_vector_run(trace, PARAMS)
    placed = {i.seq: (i.lane, i.occupancy) for i in run.insts}
    assert placed == {1: (None, 0), 2: (0, 4), 3: (1, 4), 4: (0, 4)}
    assert run.num_unverified == 0


@pytest.mark.unit
def test_flags_inconsistent_trace(tmp_path: Path) -> None:
    """An issue cycle the lane replay cannot reproduce is marked unverified."""
    trace = tmp_path / "t.tsv"
    write_trace(trace, [*ROWS[:3], (4, 11, 12, 18, "vadd.vv v12, v0, v0")])
    run = parse_vector_run(trace, PARAMS)
    assert run.num_unverified == 1
    assert not run.insts[-1].verified


@pytest.mark.unit
def test_renders_self_contained_page(tmp_path: Path) -> None:
    """The page embeds the data and needs no external script or style."""
    trace = tmp_path / "t.tsv"
    write_trace(trace, ROWS)
    page = render_html(parse_vector_run(trace, load_params()), "t")
    assert "/*DATA*/null" not in page
    assert "vadd.vv v4, v0, v0" in page
    assert "<script src" not in page
    assert "<link" not in page
