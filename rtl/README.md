# rtl

Reef RTL: the structure, the tooling, a toy design that proves lint, sim, synth and the handoff to PD end to end, and the baseline core (target `coral_m3`): CoralNPU M3 `RvvCoreMiniAxi`, generated from the Coral source in `src/reef/`.

```
src/reef/        Coral at M3-2026-04-27, our source now: Chisel in hdl/chisel/src, hand-written SV in hdl/verilog
src/reef/gen/    the SV the build uses, generated from that Chisel. Build output, never edited
src/common/      reef_defs.svh, included first by every file
src/toy/         toy_counter and its testbench, the smoke test
filelists/       one .f per target, paths relative to rtl/
targets/         one yaml per synthesis target: what to build and why
constraints/     one SDC per target, tool neutral
macros/          macro catalog: role, blackbox, views PD must supply
handoff/         the package schema and a hand-written example shaped like the core
tools/           check.sh, lint.sh, export_handoff.py, validate_handoff.py, target_args.py,
                 regen_coralnpu.sh, split_generated_sv.py, verify_upstream_release.sh
docker/          the rtl image, and the two Coral Bazel images regen uses
docs/            baseline-m3.md (the pin, flags, defines), RvvCoreMiniAxi_ports.csv (the core boundary)
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

## Coral

`src/reef/` is the M3 source, `gen/` is generated, edits go in the Chisel or the hand-written SV, CI regenerates and diffs.

- `src/reef/` is a plain copy of CoralNPU `M3-2026-04-27` (commit `72d700ef`, the pin `reef-perf/docs/deviations.md` uses), not a submodule. `docs/baseline-m3.md` lists what was copied and the one change on top so far.
- 115 of the 119 design files in `gen/` are upstream's hand-written SV (RVV backend, fpnew, `Sram.v`, `ClockGate.sv`, `RstSync.sv`); change those in `src/reef/hdl/verilog/`. Four are firtool output (`RvvCoreMiniAxi.sv`, `RvvCoreWrapper.sv`, `FloatCoreWrapper.sv`, `Aligner_119_4.sv`) and change only through the Chisel in `src/reef/hdl/chisel/src/`.
- Then run `tools/regen_coralnpu.sh` on the host (Docker, about 10 min on a cold Bazel cache) and commit `gen/` and `filelists/coral_m3.f` with the change. CI runs `tools/regen_coralnpu.sh --check` and fails on any diff.
- `tools/verify_upstream_release.sh --build` shows every file in `src/reef/` that differs from the tag, and checks that the tag's own files still build to the release asset, byte for byte.
- On macOS `src/reef/` has both `SRAM.scala` and `Sram.scala`, which collide on a case-insensitive filesystem. Use Linux or WSL to edit those two.
- SV from other repos (pe-apprentice, lambda-arch): copy the file into `src/<block>/`, add a header with the origin repo, commit and license, make it lint clean, give it a testbench beside it, add it to a filelist.

## Conventions

- One module per file, file name equals module name, snake_case. Ports `clk_i`, `rst_ni` (active low), `*_i`, `*_o`. Parameters UPPER_CASE.
- The testbench sits beside its module as `<name>_tb.sv`, self-checking, exits nonzero on failure. Filelists never list `_tb` files.
- Every file includes `reef_defs.svh` first. It sets `default_nettype none`.
- Header comment on every module: purpose, parameters, clocking and reset, owner.
- Everything in a filelist is synthesizable. Simulation-only code goes in a `_tb` file or under `ifndef SYNTHESIS`.
- Foundry cells only behind a technology-neutral wrapper module.
- `check.sh` runs Verible lint and format, Verilator `-Wall` with warnings as errors, Ruff, pytest, the toy sim and a generic Yosys synth. 2-space indent, 100 columns. `.rules.verible_lint` is the lowRISC rule set CoralNPU uses, so upstream code lints the same way here. Verilator gets the target's defines and `.rules.verilator.vlt`, which keeps `-Wall` on `src/` and waives warnings in `src/reef/` (Coral; errors still fail). Verible skips `src/reef/`.

Borrowed from how CoralNPU is laid out: tests live next to the code they test, one lint config at the root of the tree, CI builds the pinned image with a layer cache and runs everything inside it, and the deliverable is a release artifact, not a branch. Upstream attaches generated SV to each tag; here a tag `rtl-*` attaches the toy handoff package. Bazel only regenerates `gen/` from the Chisel; everything else here runs without it.

## Targets and handoff

A target is a named build: `targets/<name>.yaml` says which top, which filelist, which SDC, which defines, what the clocks are meant to be, and a status (smoke, experimental, candidate, frozen) that says how seriously to take it. It holds intent, not tool commands, so the same file drives lint, sim, synth and the export. Optional `models` lists behavioral models of the macros and foundry cells the filelist instantiates: lint and sim read them, synth-check blackboxes them, the export leaves them out.

`make export TARGET=<name>` writes `out/<name>/<sha8>/`: the RTL, filelist, SDC, io map, macro list, lint and sim reports, `manifest.yaml` and checksums. PD imports that directory, never a branch. It refuses a dirty tree unless `EXPORT_FLAGS=--allow-dirty`, and PD will not promote a dirty package past import.

`handoff/schema/handoff.schema.json` is the contract and `handoff/example/` shows the shape for the Reef core, mapped from the arch team's v0 spec. PD keeps a copy of the version number and a fixture; contract changes land here first. PD may rely on the filelist compiling with the stated defines, the SDC being tool neutral, every macro being declared, and the top and clock names being authoritative. PD gets no netlists, cells or PD constraints from here.

## Branches

Branch `rtl/<topic>` off `main`. Open a PR with the repo template, request a human reviewer, re-request after changes. Never merge your own PR, never commit to `main`.

## Open

Decisions, Alan's call:

1. The core clock period. `handoff/example` and `coral_m3` carry 20 ns as a placeholder.
2. Whether `rtl/<topic>` branch naming stands.

Known rough edges:

- Verilator cannot build under a path with a space. Outside the image, set `REEF_BUILD_DIR` to somewhere without one.
- `checks.lint: pass` in the example is what a real package must carry, not a measurement.
