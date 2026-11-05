"""Turn a timing trace into the data behind the vector-engine view.

The trace has no lane index or occupancy for vector instructions, and the
vector module does not record them. Both follow from the model, so they are
rebuilt here:

* Occupancy: `Vector::receiveInst` sets `ResultReadyCycle = IssueCycle +
  vec_latency + Occupancy - 2`, plus `vec_to_scalar_latency` for
  vector-to-scalar moves.
* Lane: `ResourcePool::reserve` books the lowest-numbered lane that frees
  first, in the order instructions reach the vector queue.

Replaying the lane bookings must reproduce every traced issue cycle. An
instruction where it does not is marked unverified rather than hidden.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from reef_perf.sim import DEFAULT_SIM_CONFIG

TEMPLATE = Path(__file__).with_name("template.html")


@dataclass(frozen=True)
class VectorParams:
    """The `top.vector.params` values the lane replay needs."""

    queue_entries: int = 8
    lanes: int = 2
    latency: int = 4
    to_scalar_latency: int = 2


@dataclass
class VectorInst:
    """One instruction that went through the vector engine."""

    seq: int
    pc: str
    disasm: str
    dispatch: int | None = None
    queued: int = 0
    issue: int = 0
    result_ready: int = 0
    complete: int | None = None
    retire: int | None = None
    lane: int | None = None  # None for vset*, which books no lane.
    occupancy: int = 0
    verified: bool = True


@dataclass
class VectorRun:
    """Vector instructions of one run plus the parameters used to place them."""

    params: VectorParams
    insts: list[VectorInst] = field(default_factory=list)
    last_cycle: int = 0

    @property
    def num_unverified(self) -> int:
        """Returns how many instructions the lane replay could not confirm."""
        return sum(not i.verified for i in self.insts)


def load_params(config: Path = DEFAULT_SIM_CONFIG) -> VectorParams:
    """Reads the vector parameters from a machine config YAML file."""
    raw = yaml.safe_load(config.read_text())["top"]["vector"]["params"]
    return VectorParams(
        queue_entries=raw["vec_queue_entries"],
        lanes=raw["vec_lanes"],
        latency=raw["vec_latency"],
        to_scalar_latency=raw["vec_to_scalar_latency"],
    )


def parse_vector_run(trace: Path, params: VectorParams) -> VectorRun:
    """Collects the vector instructions in a trace and places them on lanes."""
    by_seq: dict[int, VectorInst] = {}
    dispatch: dict[int, int] = {}
    fields: dict[int, dict[str, int]] = {}
    last_cycle = 0
    with trace.open() as f:
        next(f)  # Header.
        for line in f:
            cycle, activity, seq, pc, disasm = line.rstrip("\n").split("\t", 4)
            at, num = int(cycle), int(seq)
            last_cycle = max(last_cycle, at)
            if activity == "Vector_Buffer":
                by_seq[num] = VectorInst(num, pc, disasm, queued=at)
            elif activity == "dispatch":
                dispatch[num] = at
            elif num in by_seq:
                fields.setdefault(num, {})[activity] = at
    for num, inst in by_seq.items():
        got = fields.get(num, {})
        inst.issue = got.get("Vector", inst.queued)
        inst.result_ready = got.get("result_ready", inst.issue)
        inst.complete = got.get("complete")
        inst.retire = got.get("retire")
        inst.dispatch = dispatch.get(num)
    run = VectorRun(params, sorted(by_seq.values(), key=_queue_order))
    run.last_cycle = last_cycle
    _place_on_lanes(run)
    return run


def _queue_order(inst: VectorInst) -> tuple[int, int]:
    """Returns the order `Vector::receiveInst` sees instructions in."""
    return inst.queued, inst.seq


def _place_on_lanes(run: VectorRun) -> None:
    """Assigns lane and occupancy by replaying `ResourcePool::reserve`."""
    params = run.params
    free_at = [0] * params.lanes
    for inst in run.insts:
        if inst.disasm.startswith("vset"):
            continue
        base = inst.result_ready - inst.issue + 2 - params.latency
        # A vector-to-scalar move adds a fixed latency after the lane frees.
        for occupancy in (base, base - params.to_scalar_latency):
            lane = free_at.index(min(free_at))
            if occupancy >= 1 and max(inst.queued, free_at[lane]) == inst.issue:
                inst.lane, inst.occupancy = lane, occupancy
                free_at[lane] = inst.issue + occupancy
                break
        else:
            # Replay disagrees with the trace: trust the trace, flag it.
            inst.verified = False
            inst.occupancy = max(base, 1)
            idle = [i for i, t in enumerate(free_at) if t <= inst.issue]
            inst.lane = idle[0] if idle else free_at.index(min(free_at))
            free_at[inst.lane] = inst.issue + inst.occupancy


def render_html(run: VectorRun, title: str) -> str:
    """Returns the self-contained HTML page for the run."""
    data = {
        "title": title,
        "lanes": run.params.lanes,
        "queueEntries": run.params.queue_entries,
        "lastCycle": run.last_cycle,
        "unverified": run.num_unverified,
        "insts": [
            [
                i.seq,
                i.pc,
                i.disasm,
                i.dispatch,
                i.queued,
                i.issue,
                i.result_ready,
                i.retire,
                i.lane,
                i.occupancy,
                int(i.verified),
            ]
            for i in run.insts
        ],
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text().replace("/*DATA*/null", payload)
