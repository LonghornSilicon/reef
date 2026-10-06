"""Assemble the RISC-V workloads used by the run experiment.

Workloads are the `*.S` files in the `workloads/` directory next to this
module. They link to the Reef M3 default memory map and end with `mpause`,
so the same ELF runs on the perf model and on the M3 RTL simulator.
"""

import subprocess
from pathlib import Path

WORKLOAD_DIR = Path(__file__).resolve().parent / "workloads"
"""Directory holding the workload sources, common.inc and link.ld."""

ASSEMBLER = "riscv64-linux-gnu-as"
"""Assembler; binutils 2.38+ understands RVV 1.0."""

LINKER = "riscv64-linux-gnu-ld"
"""Linker."""

MARCH = "rv32imf_zicsr_zifencei_zve32f"
"""ISA the workloads are assembled for (Reef M3)."""


def list_workloads() -> list[str]:
    """List the available workloads.

    Returns:
        Workload names (source file stems), sorted.
    """
    return sorted(path.stem for path in WORKLOAD_DIR.glob("*.S"))


def build_workload(name: str, out_dir: Path) -> Path:
    """Assemble and link one workload.

    Args:
        name: Workload name, e.g. "alu_chain".
        out_dir: Directory for the object file and ELF.

    Returns:
        Path to the ELF.

    Raises:
        FileNotFoundError: If the workload does not exist.
        subprocess.CalledProcessError: If assembling or linking fails.
    """
    source = WORKLOAD_DIR / f"{name}.S"
    if not source.is_file():
        raise FileNotFoundError(f"no workload named {name!r} in {WORKLOAD_DIR}")
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = out_dir / f"{name}.o"
    elf = out_dir / f"{name}.elf"
    subprocess.run(
        [
            ASSEMBLER,
            f"-march={MARCH}",
            "-mabi=ilp32f",
            "-I",
            str(WORKLOAD_DIR),
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
            str(WORKLOAD_DIR / "link.ld"),
            str(obj),
            "-o",
            str(elf),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return elf


def build_workloads(names: list[str], out_dir: Path) -> dict[str, Path]:
    """Assemble and link several workloads.

    Args:
        names: Workload names.
        out_dir: Directory for the outputs.

    Returns:
        ELF path for each workload name.
    """
    return {name: build_workload(name, out_dir) for name in names}
