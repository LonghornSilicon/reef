"""Check ROI extraction from M3 RTL instruction traces."""

from pathlib import Path

import pytest

from reef_perf.experiments.correlation.rtl_trace import (
    measure_log,
    measure_roi,
    parse_trace,
)

# Abridged from a real alu_chain trace: the banner, the header, the ROI
# markers (csrr s10/s11, mcycle), two loop instructions, mpause, and the junk
# row the simulator prints after mpause.
SAMPLE = """
        SystemC 2.3.4-Accellera --- redacted redacted
        Copyright (c) 1996-2022 by all Contributors,
        ALL RIGHTS RESERVED
PC,INST,REG,DATA
0x00000014,0x3e800f93,0x1f,0x000000000000000000000000000003e8,trap=no
0x00000018,0xb0002d73,0x1a,0x0000000000000000000000000000000b,trap=no
0x0000001c,0x00b50533,0x0a,0x00000000000000000000000000000000,trap=no
0x00000060,0xfa0f9ee3,0x7f,0x00000000000000000000000000000000,trap=no
0x00000064,0xb0002df3,0x1b,0x0000000000000000000000000000465f,trap=no
0x00000068,0x08000073,0x7f,0x00000000000000000000000000000000,trap=no
0x0000006c,0x89c6ae2a,0x7f,0x00000000000000000000000000000000,trap=yes

Info: /OSCI/SystemC: Simulation stopped by user.
"""

BEGIN = "0x00000018,0xb0002d73,0x1a,0x0000000b,trap=no"
END = "0x00000064,0xb0002df3,0x1b,0x0000465f,trap=no"
ADD = "0x0000001c,0x00b50533,0x0a,0x00000000,trap=no"


@pytest.mark.unit
def test_parses_only_instruction_rows() -> None:
    """Banner, header and SystemC lines are skipped."""
    rows = parse_trace(SAMPLE)
    assert len(rows) == 7
    assert rows[1].pc == 0x18
    assert rows[1].reg == 26
    assert rows[1].data == 11
    assert rows[-1].trap


@pytest.mark.unit
def test_measures_roi() -> None:
    """Cycles are the mcycle difference; instructions exclude the markers."""
    roi = measure_roi(parse_trace(SAMPLE))
    assert roi.cycles == 0x465F - 0xB
    assert roi.retired_insts == 2
    assert roi.reached_mpause


@pytest.mark.unit
def test_measures_log_file(tmp_path: Path) -> None:
    """measure_log reads the same result from a file."""
    log = tmp_path / "alu_chain.log"
    log.write_text(SAMPLE)
    assert measure_log(log) == measure_roi(parse_trace(SAMPLE))


@pytest.mark.unit
def test_reports_missing_mpause() -> None:
    """A run that stops after ROI_END without mpause is flagged."""
    roi = measure_roi(parse_trace("\n".join([BEGIN, ADD, END])))
    assert not roi.reached_mpause


@pytest.mark.unit
def test_handles_mcycle_wraparound() -> None:
    """The low 32 bits of mcycle may wrap inside the ROI."""
    begin = "0x00000018,0xb0002d73,0x1a,0xfffffff0,trap=no"
    end = "0x00000064,0xb0002df3,0x1b,0x00000010,trap=no"
    assert measure_roi(parse_trace("\n".join([begin, end]))).cycles == 0x20


@pytest.mark.unit
def test_ignores_other_mcycle_reads() -> None:
    """Reading mcycle into a register other than s10/s11 is not a marker."""
    other = "0x00000020,0xb00025f3,0x0b,0x00000020,trap=no"  # csrr a1, mcycle
    roi = measure_roi(parse_trace("\n".join([BEGIN, other, END])))
    assert roi.retired_insts == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([ADD, END], "one ROI_BEGIN marker, found 0"),
        ([BEGIN, ADD], "one ROI_END marker, found 0"),
        ([BEGIN, BEGIN, END], "one ROI_BEGIN marker, found 2"),
        ([END, ADD, BEGIN], "ROI_END retired before ROI_BEGIN"),
    ],
)
def test_rejects_bad_markers(lines: list[str], message: str) -> None:
    """Missing, repeated or out-of-order markers are errors."""
    with pytest.raises(ValueError, match=message):
        measure_roi(parse_trace("\n".join(lines)))


@pytest.mark.unit
def test_rejects_malformed_row() -> None:
    """A row with the wrong number of fields is an error."""
    with pytest.raises(ValueError, match="malformed"):
        parse_trace("0x00000018,0xb0002d73,0x1a\n")
