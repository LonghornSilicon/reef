"""Check float-add, then the small W8A8 ELF using Coral's MPACT binding.

Run through the staged //examples/reef_mlp:validate_functional Bazel target.
The step budget bounds execution; it is not a hardware timing measurement.
"""

import argparse
import csv
import json
import struct
from collections import Counter
from pathlib import Path

import coralnpu_v2_sim_pybind as mpact
import numpy as np
from bazel_tools.tools.python.runfiles import runfiles
from elftools.elf.elffile import ELFFile

REGIONS = ((0, 8 * 1024), (0x10000, 32 * 1024))


def load(
    path: str, required: list[str]
) -> tuple[mpact.CoralNPUSimulatorPy, dict[str, tuple[int, int]]]:
    """Validate the small ELF's layout and load a fresh simulator."""
    with open(path, "rb") as stream:
        elf = ELFFile(stream)
        if elf.elfclass != 32 or not elf.little_endian:
            raise ValueError("Runner requires a little-endian ELF32")
        if elf.header["e_machine"] != "EM_RISCV":
            raise ValueError("Runner requires a RISC-V ELF")
        entry = elf.header["e_entry"]
        symtab = elf.get_section_by_name(".symtab")
        if symtab is None:
            raise ValueError("ELF has no symbol table")
        symbols = {}
        for name in required:
            matches = symtab.get_symbol_by_name(name)
            if not matches or matches[0]["st_shndx"] == "SHN_UNDEF":
                raise ValueError(f"Missing defined symbol: {name}")
            symbols[name] = (matches[0]["st_value"], matches[0]["st_size"])
        for segment in elf.iter_segments():
            if segment["p_type"] != "PT_LOAD" or not segment["p_memsz"]:
                continue
            start = segment["p_vaddr"]
            end = start + segment["p_memsz"]
            if segment["p_paddr"] != start:
                raise ValueError(
                    "Runner requires matching ELF load/runtime addresses"
                )
            if not any(
                base <= start and end <= base + size for base, size in REGIONS
            ):
                raise ValueError(
                    f"Segment [{start:#x}, {end:#x}) exceeds small TCM map"
                )
        if not 0 <= entry < REGIONS[0][1]:
            raise ValueError(f"Entry {entry:#x} outside ITCM")

    options = mpact.CoralNPUSimulatorOptions()
    options.itcm_start_address = 0
    options.itcm_length = 8 * 1024
    options.initial_misa_value = 0x40801120  # Same value as Coral's helper.
    options.exit_on_ebreak = True
    options.semihost_htif = False
    regions = []
    for index, (base, size) in enumerate(REGIONS):
        region = mpact.CoralNPUV2MemoryRegion()
        region.start_address = base
        region.length = size
        region.permissions = (
            mpact.MemoryPermission.kReadWriteExecute
            if index == 0
            else mpact.MemoryPermission.kReadWrite
        )
        regions.append(region)
    options.memory_regions = regions
    sim = mpact.CoralNPUSimulatorPy(options)
    sim.LoadProgram(path, entry)
    if sim.ReadRegister("pc") != entry:
        raise RuntimeError("Simulator did not load the ELF entry point")
    print(f"ELF={path} entry={entry:#x}", flush=True)
    for name, (address, size) in symbols.items():
        print(f"symbol {name} address={address:#x} size={size}", flush=True)
    return sim, symbols


def word(
    sim: mpact.CoralNPUSimulatorPy,
    symbols: dict[str, tuple[int, int]],
    name: str,
    *,
    signed: bool = False,
) -> int:
    """Read a checked 32-bit exported symbol."""
    address, size = symbols[name]
    if size not in (0, 4):  # _ret is a linker symbol with size zero.
        raise ValueError(f"{name} must be a 32-bit word")
    raw = sim.ReadMemory(address, 4).tobytes()
    return struct.unpack("<i" if signed else "<I", raw)[0]


def instruction_class(word: int) -> str:
    """Disjoint RV32 instruction families; fail on unsupported encodings."""
    if word & 3 != 3:
        raise ValueError("Trace classifier requires uncompressed instructions")
    opcode = word & 0x7F
    if opcode in (0x07, 0x27):
        # FP half/single/double widths differ from RVV memory width encodings.
        scalar = ((word >> 12) & 7) in (1, 2, 3)
        return ("scalar_fp_" if scalar else "vector_") + (
            "load" if opcode == 7 else "store"
        )
    classes = {
        0x03: "scalar_integer_load",
        0x23: "scalar_integer_store",
        0x13: "scalar_integer_arithmetic",
        0x33: "scalar_integer_arithmetic",
        0x17: "scalar_address_arithmetic",
        0x37: "scalar_address_arithmetic",
        0x53: "scalar_fp_arithmetic",
        0x43: "scalar_fp_arithmetic",
        0x47: "scalar_fp_arithmetic",
        0x4B: "scalar_fp_arithmetic",
        0x4F: "scalar_fp_arithmetic",
        0x57: "vector_arithmetic_or_config",
        0x63: "branch",
        0x67: "jump",
        0x6F: "jump",
        0x73: "system",
        0x0F: "fence",
    }
    if opcode not in classes:
        raise ValueError(f"Unsupported trace opcode: {word:#010x}")
    return classes[opcode]


def trace_execution(
    sim: mpact.CoralNPUSimulatorPy, max_steps: int, prefix: str
) -> tuple[int, bool]:
    """Capture Step(1) instruction words, including the terminal instruction."""
    trace_path = Path(prefix)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    total = 0
    halted = False
    with Path(str(trace_path) + ".csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ("index", "pc", "instruction_word", "instruction_class")
        )
        while total < max_steps:
            pc = sim.ReadRegister("pc")
            instruction = struct.unpack("<I", sim.ReadMemory(pc, 4).tobytes())[
                0
            ]
            category = instruction_class(instruction)
            count = sim.Step(1)
            if count != 1:
                raise RuntimeError(
                    f"Trace Step(1) did not execute one instruction: {count}"
                )
            counts[category] += 1
            total += 1
            writer.writerow((total, hex(pc), f"{instruction:#010x}", category))
            # The pinned simulator installs halt handlers for these words.
            # Step(1) returns 1 even when that instruction requests a halt, and
            # another Step call would resume it. Stop after recording the halt.
            if instruction in (0x08000073, 0x00100073):  # mpause / ebreak
                halted = True
                break
    summary = {
        "scope": (
            "whole MLP ELF, entry through terminal mpause/ebreak inclusive"
        ),
        "includes": (
            "CRT/startup, input quantization, MLP, library calls, "
            "reference checks, finalization"
        ),
        "method": (
            "PC + instruction word before each successful simulator Step(1)"
        ),
        "executed_instructions": total,
        "halt_instruction_executed": halted,
        "classes": dict(sorted(counts.items())),
        "loads": sum(
            counts[k]
            for k in ("scalar_integer_load", "scalar_fp_load", "vector_load")
        ),
        "stores": sum(
            counts[k]
            for k in ("scalar_integer_store", "scalar_fp_store", "vector_store")
        ),
        "scalar_arithmetic": sum(
            counts[k]
            for k in (
                "scalar_integer_arithmetic",
                "scalar_address_arithmetic",
                "scalar_fp_arithmetic",
            )
        ),
        "vector_instructions": sum(
            counts[k]
            for k in (
                "vector_load",
                "vector_store",
                "vector_arithmetic_or_config",
            )
        ),
        "timing": "No hardware cycle or utilization inference",
    }
    Path(str(trace_path) + ".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("instruction_mix=" + json.dumps(summary, sort_keys=True), flush=True)
    return total, halted


def execute(
    sim: mpact.CoralNPUSimulatorPy,
    symbols: dict[str, tuple[int, int]],
    max_steps: int,
    trace_prefix: str | None = None,
) -> bool:
    """Execute within the budget and report completion and return state."""
    total = 0
    halted = False
    if trace_prefix:
        total, halted = trace_execution(sim, max_steps, trace_prefix)
    while not trace_prefix and total < max_steps:
        batch = min(10000, max_steps - total)
        count = sim.Step(batch)
        if not 0 <= count <= batch:
            raise RuntimeError(f"Simulator Step failed: {count}")
        total += count
        if count < batch:
            halted = True
            break
    print(
        f"halted={halted} steps={total} "
        f"pc={sim.ReadRegister('pc'):#x} sp={sim.ReadRegister('sp'):#x} "
        f"_ret={word(sim, symbols, '_ret'):#x}",
        flush=True,
    )
    return halted


def smoke(path: str, max_steps: int) -> None:
    """Require exact float-add outputs and successful completion."""
    sim, symbols = load(path, ["input1", "input2", "output", "_ret"])
    left = np.arange(1, 9, dtype="<f4")
    right = np.full(8, np.float32(0.213), dtype="<f4")
    for name, values in (("input1", left), ("input2", right)):
        if symbols[name][1] != values.nbytes:
            raise ValueError(f"Unexpected smoke-test array size: {name}")
        raw = values.view(np.uint8)
        sim.WriteMemory(symbols[name][0], raw, raw.size)
        np.testing.assert_array_equal(
            sim.ReadMemory(symbols[name][0], raw.size), raw
        )
    halted = execute(sim, symbols, max_steps)
    if symbols["output"][1] != left.nbytes:
        raise ValueError("Unexpected smoke-test output size")
    actual = np.frombuffer(
        sim.ReadMemory(symbols["output"][0], left.nbytes), dtype="<f4"
    )
    print(f"float_add_output={actual.tolist()}", flush=True)
    np.testing.assert_array_equal(actual, left + right)
    if not halted or word(sim, symbols, "_ret") != 0:
        raise RuntimeError("Float-add did not return successfully")
    print(
        "PASS: existing Coral float-add example, 8 exact FP32 comparisons",
        flush=True,
    )


def mlp(path: str, max_steps: int, trace_prefix: str | None = None) -> None:
    """Require the original MLP checks and completion symbols to pass."""
    sim, symbols = load(
        path, ["mlp_status", "mlp_failures", "mlp_output", "_ret"]
    )
    initial = word(sim, symbols, "mlp_status", signed=True)
    print(f"loaded mlp_status={initial}", flush=True)
    if initial != -1:
        raise RuntimeError("Expected the ELF's incomplete sentinel (-1)")
    halted = execute(sim, symbols, max_steps, trace_prefix)
    status = word(sim, symbols, "mlp_status", signed=True)
    failures = word(sim, symbols, "mlp_failures")
    print(f"mlp_status={status} mlp_failures={failures}", flush=True)
    if symbols["mlp_output"][1] != 3 * 8 * 4:
        raise ValueError("Runner is restricted to the 3-token 8->16->8 fixture")
    output = np.frombuffer(
        sim.ReadMemory(symbols["mlp_output"][0], 96), dtype="<f4"
    )
    print(f"mlp_output={output.tolist()}", flush=True)
    if (
        not halted
        or status != 0
        or failures != 0
        or word(sim, symbols, "_ret") != 0
    ):
        raise RuntimeError(
            "MLP incomplete, trapped, timed out, or failed its checks"
        )
    print(
        "PASS: MLP exact integer checks and existing FP32 tolerances",
        flush=True,
    )


def main() -> None:
    """Resolve Bazel runfiles and validate float-add before the small MLP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument(
        "--trace-prefix",
        help="Write whole-MLP-ELF executed trace CSV and instruction-mix JSON",
    )
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.smoke_only and args.trace_prefix:
        parser.error("--trace-prefix applies to the MLP run; omit --smoke-only")
    resolver = runfiles.Create()

    def resolve(relative: str) -> str:
        """Locate a required ELF in this target's Bazel runfiles."""
        path = resolver.Rlocation("coralnpu_hw/" + relative)
        if not path:
            raise FileNotFoundError(relative)
        return path

    smoke(
        resolve("examples/coralnpu_v2_hello_world_add_floats.elf"),
        args.max_steps,
    )
    if not args.smoke_only:
        mlp(
            resolve("examples/reef_mlp/mlp_w8a8_test.elf"),
            args.max_steps,
            args.trace_prefix,
        )


if __name__ == "__main__":
    main()
