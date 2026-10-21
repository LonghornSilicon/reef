"""Read region-of-interest cycle counts out of M3 RTL instruction traces.

The trace is what the CoralNPU verification simulator prints with
`--instr_trace` (`rvv_core_mini_verification_axi_sim` at tag
`M3-2026-04-27`): a few banner lines, a `PC,INST,REG,DATA` header, then one
row per retired instruction, in retirement order:

    0x00000018,0xb0002d73,0x1a,0x...0000000b,trap=no

REG is the destination register index (x0-x31, f regs from 32, v regs from
64; 0x7f means none) and DATA is the 128-bit value written.

Every microbenchmark brackets its loop with `csrr s10, mcycle` (ROI_BEGIN) and
`csrr s11, mcycle` (ROI_END), so the loop's RTL cycle count is the difference
of the two values written.
"""

from dataclasses import dataclass
from pathlib import Path

MCYCLE = 0xB00
"""CSR address of mcycle."""

ROI_BEGIN_REG = 26
"""s10 (x26): written by `csrr s10, mcycle` at ROI_BEGIN."""

ROI_END_REG = 27
"""s11 (x27): written by `csrr s11, mcycle` at ROI_END."""

MPAUSE = 0x08000073
"""Encoding of mpause, the instruction every microbenchmark ends with."""

_SYSTEM_OPCODE = 0x73
_CSRRS_FUNCT3 = 0b010
_COUNTER_MASK = (1 << 32) - 1


@dataclass(frozen=True)
class TraceRow:
    """One retired instruction.

    Attributes:
        pc: Program counter.
        inst: Instruction encoding.
        reg: Destination register index, or 0x7f for none.
        data: Value written to the destination register.
        trap: Whether the instruction trapped.
    """

    pc: int
    inst: int
    reg: int
    data: int
    trap: bool


@dataclass(frozen=True)
class RoiMeasurement:
    """What one trace says about the region of interest.

    Attributes:
        cycles: mcycle at ROI_END minus mcycle at ROI_BEGIN.
        retired_insts: Instructions retired strictly between the two markers.
        reached_mpause: Whether the program retired mpause after ROI_END.
    """

    cycles: int
    retired_insts: int
    reached_mpause: bool


def parse_trace(text: str) -> list[TraceRow]:
    """Parse the instruction rows of a trace, skipping everything else.

    Args:
        text: The simulator's output.

    Returns:
        The retired instructions, in retirement order.

    Raises:
        ValueError: If a row starting with `0x` is malformed.
    """
    rows = []
    for line in text.splitlines():
        if not line.startswith("0x"):
            continue  # banner, header, SystemC messages
        fields = line.strip().split(",")
        if len(fields) != 5 or not fields[4].startswith("trap="):
            raise ValueError(f"malformed trace row: {line!r}")
        rows.append(
            TraceRow(
                pc=int(fields[0], 16),
                inst=int(fields[1], 16),
                reg=int(fields[2], 16),
                data=int(fields[3], 16),
                trap=fields[4] == "trap=yes",
            )
        )
    return rows


def reads_mcycle_into(row: TraceRow, reg: int) -> bool:
    """Check whether a row is `csrr x<reg>, mcycle`.

    `csrr rd, csr` is `csrrs rd, csr, x0`.

    Args:
        row: A retired instruction.
        reg: Destination register index.

    Returns:
        True if the row reads mcycle into register `reg`.
    """
    inst = row.inst
    return (
        inst & 0x7F == _SYSTEM_OPCODE
        and (inst >> 12) & 0x7 == _CSRRS_FUNCT3
        and (inst >> 15) & 0x1F == 0
        and (inst >> 20) == MCYCLE
        and (inst >> 7) & 0x1F == reg
        and row.reg == reg
    )


def _find_marker(rows: list[TraceRow], reg: int, name: str) -> int:
    """Find the one row that reads mcycle into `reg`.

    Args:
        rows: The retired instructions.
        reg: Destination register of the marker.
        name: Marker name, for error messages.

    Returns:
        Index of the marker row.

    Raises:
        ValueError: If there is not exactly one such row.
    """
    hits = [i for i, row in enumerate(rows) if reads_mcycle_into(row, reg)]
    if len(hits) != 1:
        raise ValueError(f"expected one {name} marker, found {len(hits)}")
    return hits[0]


def measure_roi(rows: list[TraceRow]) -> RoiMeasurement:
    """Measure the region of interest between ROI_BEGIN and ROI_END.

    Args:
        rows: The retired instructions of one run.

    Returns:
        The ROI cycle count, instruction count and whether mpause retired.

    Raises:
        ValueError: If either marker is missing or repeated, or ROI_END
            retires before ROI_BEGIN.
    """
    begin = _find_marker(rows, ROI_BEGIN_REG, "ROI_BEGIN")
    end = _find_marker(rows, ROI_END_REG, "ROI_END")
    if end < begin:
        raise ValueError("ROI_END retired before ROI_BEGIN")
    # csrr reads the low 32 bits of mcycle; difference them modulo 2^32.
    cycles = (rows[end].data - rows[begin].data) & _COUNTER_MASK
    return RoiMeasurement(
        cycles=cycles,
        retired_insts=end - begin - 1,
        reached_mpause=any(row.inst == MPAUSE for row in rows[end + 1 :]),
    )


def measure_log(path: Path) -> RoiMeasurement:
    """Measure the region of interest in a trace file.

    Args:
        path: A log written by the simulator with `--instr_trace`.

    Returns:
        The ROI measurement.

    Raises:
        ValueError: If the trace is malformed or the markers are wrong.
    """
    return measure_roi(parse_trace(Path(path).read_text()))
