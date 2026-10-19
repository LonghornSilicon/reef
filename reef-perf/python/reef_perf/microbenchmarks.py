"""Assemble the RISC-V microbenchmarks.

The microbenchmarks are the `*.S` files in reef-perf/microbenchmarks/, shared
by the tests and the experiments. They link to the Reef M3 default memory map
and end with `mpause`, so the same ELF runs on the perf model and on the M3 RTL
simulator. Each brackets its loop with `csrr s10, mcycle` / `csrr s11, mcycle`,
so both report the same region-of-interest cycle count.
"""

import subprocess
from pathlib import Path

from reef_perf.sim import MODULE_ROOT

MICROBENCHMARK_DIR = MODULE_ROOT / "microbenchmarks"
"""Directory holding the microbenchmark sources, common.inc and link.ld."""

ASSEMBLER = "riscv64-linux-gnu-as"
"""Assembler; binutils 2.38+ understands RVV 1.0."""

LINKER = "riscv64-linux-gnu-ld"
"""Linker."""

MARCH = "rv32imf_zicsr_zifencei_zve32f"
"""ISA the microbenchmarks are assembled for (Reef M3)."""


def list_microbenchmarks() -> list[str]:
    """List the available microbenchmarks.

    Returns:
        Microbenchmark names (source file stems), sorted.
    """
    return sorted(path.stem for path in MICROBENCHMARK_DIR.glob("*.S"))


def build_microbenchmark(name: str, out_dir: Path) -> Path:
    """Assemble and link one microbenchmark.

    Args:
        name: Microbenchmark name, e.g. "alu_chain".
        out_dir: Directory for the object file and ELF.

    Returns:
        Path to the ELF.

    Raises:
        FileNotFoundError: If the microbenchmark does not exist.
        subprocess.CalledProcessError: If assembling or linking fails.
    """
    source = MICROBENCHMARK_DIR / f"{name}.S"
    if not source.is_file():
        raise FileNotFoundError(
            f"no microbenchmark named {name!r} in {MICROBENCHMARK_DIR}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = out_dir / f"{name}.o"
    elf = out_dir / f"{name}.elf"
    subprocess.run(
        [
            ASSEMBLER,
            f"-march={MARCH}",
            "-mabi=ilp32f",
            "-I",
            str(MICROBENCHMARK_DIR),
            str(source),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            LINKER,
            "-m",
            "elf32lriscv",
            "-T",
            str(MICROBENCHMARK_DIR / "link.ld"),
            str(obj),
            "-o",
            str(elf),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return elf


def build_microbenchmarks(names: list[str], out_dir: Path) -> dict[str, Path]:
    """Assemble and link several microbenchmarks.

    Args:
        names: Microbenchmark names.
        out_dir: Directory for the outputs.

    Returns:
        ELF path for each microbenchmark name.
    """
    return {name: build_microbenchmark(name, out_dir) for name in names}
