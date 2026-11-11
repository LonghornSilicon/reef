"""A cycle-level reference for the matrix fabric's non-Coral arrays.

The matrix model times systolic and adder-tree arrays with closed-form fill,
drain and stationary-load terms (MatrixFabric.cpp; docs/matrix.md, case 17).
No RTL has those arrays, so this module checks the formulas against an
independent, cycle-by-cycle schedule of the same hardware.

For each pass of a MAC (one array-sized block of the tile), the reference
places every event on a cycle: each operand's arrival at each PE, each
multiply, each partial-sum hop and each stationary-operand load. It derives
arrival times from where operands enter the array and how they move
(forwarded one PE per hop, or broadcast), independently for the two
operands, and fails if the two operands of a multiply do not meet in the
same cycle. It books every PE port and register per cycle and starts each
pass at the earliest cycle at which nothing collides. A run also checks that
every product of the tile is computed exactly once.

The hardware it schedules, per dataflow:

- output: C stays in each PE (in place); A enters each row and B each
  column; a PE multiplies `lanes` K elements per cycle.
- weight: B stays; rows hold K (`lanes` elements per PE), columns hold N, M
  streams in time. A enters each row; partial sums move down a column one PE
  per hop (chain) or go to a per-column adder tree (tree).

Stationary operands load one PE row per cycle through the column (or in
`stationary_load_cycles`), into a shadow register when double buffered.
Only the dataflows and deliveries the presets use are modeled; input
stationary is not.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from math import ceil, log2


@dataclass(frozen=True)
class Array:
    """One array configuration, named after the model's mtx_* parameters.

    Attributes:
        rows: mtx_pe_rows.
        cols: mtx_pe_cols.
        lanes: MACs per PE per cycle for the input type.
        dataflow: "output" or "weight".
        row_operand: "broadcast" or "forwarded" (A, entering each row).
        col_operand: "broadcast" or "forwarded" (B for output-stationary;
            for weight-stationary, how the partial-sum wavefront is skewed).
        reduction: "in_place", "chain" or "tree".
        hop: mtx_hop_latency.
        tree_level: mtx_tree_level_latency.
        depth: mtx_pe_pipeline_depth.
        stationary_load_cycles: 0 for one PE row per cycle.
        double_buffer: mtx_double_buffer_stationary.
    """

    rows: int
    cols: int
    lanes: int
    dataflow: str
    row_operand: str
    col_operand: str
    reduction: str
    hop: int = 1
    tree_level: int = 1
    depth: int = 1
    stationary_load_cycles: int = 0
    double_buffer: bool = False


@dataclass
class Pass:
    """One array-sized block of a MAC, with the tile indices it covers."""

    op: int
    rows: list[int]  # Tile index on each used PE row (M or K group).
    cols: list[int]  # Tile index on each used PE column (N).
    steps: int  # Temporal steps (K groups for output, M for weight).


@dataclass
class Booking:
    """Everything booked so far, per (resource, PE or column, cycle)."""

    used: set[tuple] = field(default_factory=set)
    # Stationary registers: (row, col, buffer) -> last cycle in use.
    stationary_busy_until: dict[tuple[int, int, int], int] = field(
        default_factory=dict
    )

    def free(self, keys: list[tuple]) -> bool:
        """Whether no key is booked yet and none repeats."""
        return all(k not in self.used for k in keys) and len(set(keys)) == len(
            keys
        )


@dataclass
class OpTiming:
    """When one MAC started and when its last result was available."""

    start: int
    done: int


def _passes(arr: Array, op: int, m: int, n: int, k: int) -> list[Pass]:
    kgroups = ceil(k / arr.lanes)
    if arr.dataflow == "output":
        spatial_rows, spatial_cols, steps = m, n, kgroups
    elif arr.dataflow == "weight":
        spatial_rows, spatial_cols, steps = kgroups, n, m
    else:
        raise ValueError(f"dataflow {arr.dataflow!r} is not modeled")
    out = []
    for r0 in range(0, spatial_rows, arr.rows):
        for c0 in range(0, spatial_cols, arr.cols):
            out.append(
                Pass(
                    op,
                    list(range(r0, min(r0 + arr.rows, spatial_rows))),
                    list(range(c0, min(c0 + arr.cols, spatial_cols))),
                    steps,
                )
            )
    return out


def _arrival(delivery: str, enter: int, hops: int, hop: int) -> int:
    """Cycle an operand entering at `enter` reaches a PE `hops` PEs in."""
    return enter + (hops * hop if delivery == "forwarded" else 0)


def _events(
    arr: Array, p: Pass, s: int
) -> Iterator[tuple[int, str, int, int, int]]:
    """Yields (cycle, kind, row, col, step) for pass `p` started at `s`.

    Operands enter with the skew their delivery needs: a forwarded operand
    that meets one moving the other way enters later by the other's hops.
    """
    nrows, ncols = len(p.rows), len(p.cols)
    for t in range(p.steps):
        for i in range(nrows):
            for j in range(ncols):
                if arr.dataflow == "output":
                    # A row i enters at column 0, skewed by B's travel down.
                    a_enter = s + t + _arrival(arr.col_operand, 0, i, arr.hop)
                    b_enter = s + t + _arrival(arr.row_operand, 0, j, arr.hop)
                    a = _arrival(arr.row_operand, a_enter, j, arr.hop)
                    b = _arrival(arr.col_operand, b_enter, i, arr.hop)
                else:  # weight: B is in place; A must meet the psum wave.
                    if arr.reduction == "chain":
                        # The partial sum for (t, column j) reaches row i
                        # i hops after row 0.
                        wave = s + t + i * arr.hop
                    else:
                        wave = s + t
                    a_enter = wave
                    a = _arrival(arr.row_operand, a_enter, j, arr.hop)
                    b = a  # Stationary: present whenever A arrives.
                    # The wavefront at (i, j) must follow A along the row.
                    if arr.reduction == "chain":
                        wave_at = (
                            s
                            + t
                            + i * arr.hop
                            + (
                                j * arr.hop
                                if arr.row_operand == "forwarded"
                                else 0
                            )
                        )
                        if a != wave_at:
                            raise AssertionError("A misses the partial sum")
                if a != b:
                    raise AssertionError(
                        f"operands miss at PE({i},{j}) step {t}: {a} vs {b}"
                    )
                yield (a, "mac", i, j, t)


def _done_cycle(arr: Array, p: Pass, last_mac: dict[int, int]) -> int:
    """Cycle the pass's last result is available, from per-column last MACs."""
    nrows = len(p.rows)
    done = 0
    inputs = nrows * arr.lanes
    for t_mac in last_mac.values():
        if arr.reduction == "tree":
            levels = ceil(log2(inputs)) if inputs > 1 else 0
            t_out = t_mac + levels * arr.tree_level
        else:
            t_out = t_mac  # Chain: the bottom row's MAC is the last add.
        # As in the model, depth counts from the cycle after the last step.
        done = max(done, t_out + 1 + arr.depth)
    return done


def schedule(arr: Array, shapes: list[tuple[int, int, int]]) -> list[OpTiming]:
    """Schedules independent MACs back to back, each pass as early as legal.

    Args:
        arr: The array.
        shapes: (M, N, K) per MAC, in issue order.

    Returns:
        Each MAC's start (first load or operand cycle) and done cycle.
    """
    book = Booking()
    timings: list[OpTiming] = []
    covered: dict[int, set[tuple[int, int, int]]] = {}
    pass_index = 0
    prev_start = 0
    for op, (m, n, k) in enumerate(shapes):
        op_start = None
        op_done = 0
        for p in _passes(arr, op, m, n, k):
            buf = pass_index % 2 if arr.double_buffer else 0
            pass_index += 1
            load = 0
            if arr.dataflow == "weight":
                load = arr.stationary_load_cycles or len(p.rows)
            s = prev_start
            while True:
                # Each column loads its stationary operands into `buf` in
                # the `load` cycles before A first reaches that column (in
                # step with the wavefront), once that buffer's last use in
                # the column has passed.
                load_start = s - load
                ok = load_start >= 0
                skew = [
                    j * arr.hop if arr.row_operand == "forwarded" else 0
                    for j in range(len(p.cols))
                ]
                if ok and load:
                    ok = all(
                        s + skew[j] - load
                        > book.stationary_busy_until.get((i, j, buf), -1)
                        for i in range(len(p.rows))
                        for j in range(len(p.cols))
                    )
                if ok:
                    events = list(_events(arr, p, s))
                    keys = [("pe", i, j, c) for c, _, i, j, _ in events]
                    # The load path down each column carries one load.
                    keys += [
                        ("load", j, c)
                        for j in range(len(p.cols))
                        for c in range(s + skew[j] - load, s + skew[j])
                    ]
                    if arr.reduction == "tree":
                        # One input set per column per cycle into its tree.
                        keys += list(
                            {("tree", j, c) for c, _, _, j, _ in events}
                        )
                    ok = book.free(keys)
                if ok:
                    break
                s += 1
            book.used.update(keys)
            last_mac: dict[int, int] = {}
            for c, _, i, j, t in events:
                last_mac[j] = max(last_mac.get(j, 0), c)
                if arr.dataflow == "weight":
                    key = (i, j, buf)
                    book.stationary_busy_until[key] = max(
                        book.stationary_busy_until.get(key, -1), c
                    )
                for lane in range(arr.lanes):
                    if arr.dataflow == "output":
                        mi, ni, ki = p.rows[i], p.cols[j], t * arr.lanes + lane
                    else:
                        mi, ni, ki = t, p.cols[j], p.rows[i] * arr.lanes + lane
                    if ki < k:
                        trip = (mi, ni, ki)
                        seen = covered.setdefault(op, set())
                        if trip in seen:
                            raise AssertionError(f"product {trip} twice")
                        seen.add(trip)
            start = s - load
            op_start = start if op_start is None else min(op_start, start)
            op_done = max(op_done, _done_cycle(arr, p, last_mac))
            prev_start = s
        if len(covered[op]) != m * n * k:
            raise AssertionError(f"op {op}: {len(covered[op])} of {m * n * k}")
        timings.append(OpTiming(op_start, op_done))
    return timings


def measure(
    arr: Array, m: int, n: int, k: int, ops: int = 6
) -> dict[str, float]:
    """Latency of one MAC alone and the interval between back-to-back MACs.

    Returns:
        latency: cycles from a lone MAC's start to its last result.
        interval: steady-state cycles between independent MACs' starts.
    """
    alone = schedule(arr, [(m, n, k)])[0]
    stream = schedule(arr, [(m, n, k)] * ops)
    interval = (stream[-1].start - stream[1].start) / (ops - 2)
    return {"latency": alone.done - alone.start, "interval": interval}
