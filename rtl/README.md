# rtl

Reef RTL: the structure, the tooling, a toy design that proves lint, sim, synth and the handoff to PD end to end, and Drop 0 (target `reef_top_sky130_sram22`), the CoralNPU M3 core with the Reef top and sky130 wrappers.

```
src/reef/        Reef-owned SV, split by block: top/ (reef_top), sky130/ (macro and cell wrappers)
src/common/      reef_defs.svh, included first by every file
src/toy/         toy_counter and its testbench, the smoke test
ext/coralnpu     CoralNPU M3-2026-04-27 as a git submodule, a pointer only
ext/coralnpu-sv  the generated SV of that release the build uses, see its README
ext/sram22       SRAM22 macro model, lint and sim only
filelists/       one .f per target, paths relative to rtl/
targets/         one yaml per synthesis target: what to build and why
constraints/     one SDC per target, tool neutral
macros/          macro catalog: role, blackbox, views PD must supply
handoff/         the package schema and a hand-written example shaped like the core
tools/           check.sh, lint.sh, export_handoff.py, validate_handoff.py, target_args.py
```

## Run

Everything runs in the image.

```
rtl/docker/run.sh build
rtl/docker/run.sh shell
make check                       # what CI runs
make lint                        # auto-fix formatting, then check
make sim TARGET=toy_counter
make synth-check TARGET=toy_counter
make export TARGET=toy_counter   # writes out/toy_counter/<sha8>/
uv run python tools/validate_handoff.py out/toy_counter/*
```

Pinned in `docker/Dockerfile`: OSS CAD Suite 2026-10-12 (Verilator 5.053, Yosys 0.69), Verible v0.0-4296, uv 0.10.10, Ruff 0.16.5.

## Existing RTL

- The baseline core is CoralNPU release `M3-2026-04-27`, commit `72d700ef`, the same pin `reef-perf/docs/deviations.md` uses. `ext/coralnpu` is a submodule pointing at that commit and is empty until you run `git submodule update --init rtl/ext/coralnpu`. Nothing runs it for you. Never edit anything under `ext/`.
- The submodule holds Chisel, not SystemVerilog. The SV to synthesize is the `RvvCoreMiniAxi.sv` asset Google attaches to the release tag. It is in `ext/coralnpu-sv/`, split one file per module; its README has the tag, the sha256 and the one patch on top.
- On macOS the submodule shows as modified forever: upstream has both `SRAM.scala` and `Sram.scala`, which collide on a case-insensitive filesystem. Harmless. The export tool ignores submodule contents for that reason.
- SV from other repos (pe-apprentice, lambda-arch): copy the file into `src/reef/`, add a header with the origin repo, commit and license, make it lint clean, give it a testbench in `tb/unit/`, add it to a filelist.

## Conventions

- One module per file, file name equals module name, snake_case. Ports `clk_i`, `rst_ni` (active low), `*_i`, `*_o`. Parameters UPPER_CASE.
- The testbench sits beside its module as `<name>_tb.sv`, self-checking, exits nonzero on failure. Filelists never list `_tb` files.
- Every file includes `reef_defs.svh` first. It sets `default_nettype none`.
- Header comment on every module: purpose, parameters, clocking and reset, owner.
- Everything in a filelist is synthesizable. Simulation-only code goes in a `_tb` file or under `ifndef SYNTHESIS`.
- Foundry cells only behind a technology-neutral wrapper module.
- `check.sh` runs Verible lint and format, Verilator `-Wall` with warnings as errors, Ruff, pytest, the toy sim and a generic Yosys synth. 2-space indent, 100 columns. `.rules.verible_lint` is the lowRISC rule set CoralNPU uses, so upstream code lints the same way here. Verilator gets the target's defines and `.rules.verilator.vlt`, which keeps `-Wall` on `src/` and waives warnings in `ext/` (upstream code; errors still fail).

Borrowed from how CoralNPU is laid out: tests live next to the code they test, one lint config at the root of the tree, CI builds the pinned image with a layer cache and runs everything inside it, and the deliverable is a release artifact, not a branch. Upstream attaches generated SV to each tag; here a tag `rtl-*` attaches the toy handoff package. Not borrowed: Bazel, which is more build system than a student team needs.

## Targets and handoff

A target is a named build: `targets/<name>.yaml` says which top, which filelist, which SDC, which defines, what the clocks are meant to be, and a status (smoke, experimental, candidate, frozen) that says how seriously to take it. It holds intent, not tool commands, so the same file drives lint, sim, synth and the export. Optional `models` lists behavioral models of the macros and foundry cells the filelist instantiates: lint and sim read them, synth-check blackboxes them, the export leaves them out.

`make export TARGET=<name>` writes `out/<name>/<sha8>/`: the RTL, filelist, SDC, io map, macro list, lint and sim reports, `manifest.yaml` and checksums. PD imports that directory, never a branch. It refuses a dirty tree unless `EXPORT_FLAGS=--allow-dirty`, and PD will not promote a dirty package past import.

`handoff/schema/handoff.schema.json` is the contract and `handoff/example/` shows the shape for the Reef core, mapped from the arch team's v0 spec. PD keeps a copy of the version number and a fixture; contract changes land here first. PD may rely on the filelist compiling with the stated defines, the SDC being tool neutral, every macro being declared, and the top and clock names being authoritative. PD gets no netlists, cells or PD constraints from here.

## Branches

Branch `rtl/<topic>` off `main`. Open a PR with the repo template, request a human reviewer, re-request after changes. Never merge your own PR, never commit to `main`.

## Open

Decisions, Alan's call:

1. How CoralNPU comes in: submodule, vendored generated SV, or Chisel kept as the generator.
2. The core clock period. `handoff/example` carries 20 ns as a placeholder.
3. Whether `rtl/<topic>` branch naming stands.

Known rough edges:

- `macros/sram22.yaml` has `blackbox: TBD`, so the export ships no blackbox for the SRAM22 macro yet.
- Verilator cannot build under a path with a space. Outside the image, set `REEF_BUILD_DIR` to somewhere without one.
- `checks.lint: pass` in the example is what a real package must carry, not a measurement.
