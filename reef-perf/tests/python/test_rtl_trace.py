
import pytest

from reef_perf.experiments.correlate.rtl_trace import parse_rtl_trace

pytestmark = pytest.mark.unit

def test_parse_rtl_trace(tmp_path):
    trace = tmp_path / "sample.log"

    trace.write_text(
        "Simulator starting...\n"
        "0x00000018,0xb0002d73,0x1a,0x0000000000000000000000000000000b,trap=no\n"
        "0x0000001c,0x00100093,0x01,0x00000000000000000000000000000001,trap=no\n"
        "0x00000020,0x00200113,0x02,0x00000000000000000000000000000002,trap=no\n"
        "0x00000024,0xb0002df3,0x1b,0x00000000000000000000000000000015,trap=no\n"
    )

    result = parse_rtl_trace(trace)

    assert result.start_cycle == 11
    assert result.end_cycle == 21
    assert result.roi_cycles == 10
    assert result.retired_insts == 2


def test_missing_end_marker(tmp_path):
    trace = tmp_path / "missing_end.log"

    trace.write_text(
        "0x00000018,0xb0002d73,0x1a,0x0000000000000000000000000000000b,trap=no\n"
    )

    with pytest.raises(ValueError, match="ROI end marker not found"):
        parse_rtl_trace(trace)
