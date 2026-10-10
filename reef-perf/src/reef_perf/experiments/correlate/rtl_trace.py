
"""Parse CoralNPU M3 RTL instruction traces and extract ROI metrics."""

from dataclasses import dataclass
from pathlib import Path


MCYCLE_CSR = 0xB00
START_REGISTER = 26  # s10
END_REGISTER = 27    # s11


@dataclass(frozen=True)
class RTLTraceResult:
    roi_cycles: int
    retired_insts: int
    start_cycle: int
    end_cycle: int


def parse_rtl_trace(path: str | Path) -> RTLTraceResult:
    """Extract ROI cycles and retired instructions from an RTL trace.

    The ROI starts at `csrr s10, mcycle` and ends at
    `csrr s11, mcycle`. Both marker instructions are excluded
    from the retired instruction count.
    """
    start_cycle = None
    end_cycle = None
    retired_insts = 0

    with open(path, "r", encoding="utf-8") as trace:
        for line in trace:
            fields = line.strip().split(",")

            # Ignore simulator messages and malformed trace lines.
            if len(fields) != 5:
                continue

            try:
                instruction = int(fields[1], 16)
                destination = int(fields[2], 16)
                value = int(fields[3], 16)
            except ValueError:
                continue

            # Identify CSR read instructions by decoding their bits.
            opcode = instruction & 0x7F
            funct3 = (instruction >> 12) & 0x7
            csr = (instruction >> 20) & 0xFFF

            is_mcycle_read = (
                opcode == 0x73
                and funct3 == 0x2  # CSRRS (csrr pseudo-instruction)
                and csr == MCYCLE_CSR
                and ((instruction >> 15) & 0x1F) == 0
            )

            if is_mcycle_read and destination == START_REGISTER:
                if start_cycle is not None:
                    raise ValueError("Duplicate ROI start marker")

                start_cycle = value
                continue

            if is_mcycle_read and destination == END_REGISTER:
                if start_cycle is None:
                    raise ValueError("ROI end marker found before start marker")

                end_cycle = value
                break

            if start_cycle is not None:
                retired_insts += 1

    if start_cycle is None:
        raise ValueError("ROI start marker not found")

    if end_cycle is None:
        raise ValueError("ROI end marker not found")

    if end_cycle < start_cycle:
        raise ValueError("ROI end cycle precedes start cycle")

    return RTLTraceResult(
        roi_cycles=end_cycle - start_cycle,
        retired_insts=retired_insts,
        start_cycle=start_cycle,
        end_cycle=end_cycle,
    )
