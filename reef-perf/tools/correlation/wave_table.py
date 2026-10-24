"""Print RTL waveform signals as a table with one row per clock cycle.

Reads an FST file from the M3 Verilator simulator (run with --trace) through
the fstr reader (fstr.c in this directory), samples the chosen signals at
each rising edge of the core clock and prints the cycles asked for. Cycle 0
is the first rising edge. Values are hexadecimal; x and z bits print as ?.

Usage:

    python3 wave_table.py core.fst --list rd_valid_rob2rt
    python3 wave_table.py core.fst --first 60 --last 90 \
        -s dispatch_pc=score.dispatch.io_inst_0_bits_addr \
        -s rob2rt=backend.rd_valid_rob2rt_o
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CLOCK = "core.RvvCoreMiniVerificationAxi.core.score.clock"


@dataclass(frozen=True)
class Signal:
    """One FST variable.

    Attributes:
        handle: The FST handle; aliases share one.
        name: The full hierarchical name.
    """

    handle: int
    name: str


def list_signals(fstr: str, fst: Path) -> list[Signal]:
    """Return every variable in an FST file."""
    out = subprocess.run(
        [fstr, "list", str(fst)], capture_output=True, text=True, check=True
    ).stdout
    signals = []
    for line in out.splitlines():
        handle, _, name = line.split(" ", 2)
        # Drop the bit range Verilator appends, e.g. " [3:0]".
        signals.append(Signal(int(handle), name.split(" [")[0]))
    return signals


def resolve(signals: list[Signal], name: str) -> Signal:
    """Find the signal with this full name, or the only one ending in it.

    Raises:
        SystemExit: If no signal or more than one distinct signal matches.
    """
    exact = [s for s in signals if s.name == name]
    if exact:
        return exact[0]
    matches = {s.handle: s for s in signals if s.name.endswith("." + name)}
    if len(matches) != 1:
        found = ", ".join(s.name for s in list(matches.values())[:5])
        sys.exit(f"{name!r} matches {len(matches)} signals: {found}")
    return next(iter(matches.values()))


def to_hex(bits: str) -> str:
    """Return a binary string in hexadecimal, or ? if a bit is x or z."""
    if any(b not in "01" for b in bits):
        return "?"
    return f"{int(bits, 2):x}"


def sample(
    fstr: str, fst: Path, clock: Signal, columns: list[Signal]
) -> list[dict[int, str]]:
    """Return the value of every column during each clock cycle.

    A cycle's values are the ones held just before the rising edge that ends
    it.
    """
    handles = sorted({clock.handle, *(s.handle for s in columns)})
    out = subprocess.run(
        [fstr, "dump", str(fst), ",".join(map(str, handles))],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    changes: dict[int, list[tuple[int, str]]] = {}
    for line in out.splitlines():
        time, handle, value = line.split()
        changes.setdefault(int(time), []).append((int(handle), value))

    current = dict.fromkeys(handles, "0")
    cycles = []
    for time in sorted(changes):
        updates = dict(changes[time])
        if updates.get(clock.handle) == "1" and current[clock.handle] == "0":
            cycles.append(dict(current))
        current.update(updates)
    return cycles


def main() -> int:
    """Parse the arguments and print the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fst", type=Path, help="FST file from --trace")
    parser.add_argument(
        "-s",
        "--signal",
        action="append",
        default=[],
        metavar="LABEL=NAME",
        help="column to print; NAME is a full name or a unique suffix",
    )
    parser.add_argument("--first", type=int, default=0, help="first cycle")
    parser.add_argument("--last", type=int, help="last cycle (default: end)")
    parser.add_argument("--clock", default=DEFAULT_CLOCK, help="clock signal")
    parser.add_argument(
        "--list", metavar="TEXT", help="list signals containing TEXT and exit"
    )
    parser.add_argument(
        "--fstr",
        default=os.environ.get("FSTR", "fstr"),
        help="fstr executable (default: $FSTR or fstr on PATH)",
    )
    args = parser.parse_args()

    signals = list_signals(args.fstr, args.fst)
    if args.list is not None:
        for s in signals:
            if args.list in s.name:
                print(s.handle, s.name)
        return 0

    labels = []
    columns = []
    for item in args.signal:
        label, _, name = item.partition("=")
        labels.append(label)
        columns.append(resolve(signals, name or label))
    cycles = sample(args.fstr, args.fst, resolve(signals, args.clock), columns)

    last = len(cycles) - 1 if args.last is None else args.last
    widths = [max(len(label), 8) for label in labels]
    print(
        "cycle "
        + " ".join(f"{lb:>{w}}" for lb, w in zip(labels, widths, strict=True))
    )
    for n in range(args.first, min(last, len(cycles) - 1) + 1):
        values = [to_hex(cycles[n][s.handle]) for s in columns]
        print(
            f"{n:5d} "
            + " ".join(f"{v:>{w}}" for v, w in zip(values, widths, strict=True))
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
