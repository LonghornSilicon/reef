# pd

Physical design for Reef. Open flow on SKY130 by default, shaped so a private node drops in without restructuring. This is scaffolding: the stage structure, config layering, handoff import, private guard and CI are real; the stage scripts that call synthesis and place and route are stubs.

```
flow/          stages.mk (stage rules), run_stage.sh (stage contract), config.py (config layers), util.tcl (shared procs)
flow/stages/   00_import 10_synth 20_floorplan 30_place 40_cts 50_route 60_signoff 70_export
tech/sky130/   platform.yaml and dont_use.txt, values from the installed PDK via ${PDK_ROOT}
tech/private/  empty slot for a private node, mounted at run time
designs/       one design.yaml per target: toy_counter, reef_top_sky130_openframe
inputs/        symlinks to RTL handoff packages, never committed
runs/          runs/<target>/<run_id>/<stage>/{logs,reports,results}, never committed
tests/         fixtures/handoff_example (a real toy_counter export) and the python tests
tools/         check.sh, check_no_private.sh, private_denylist.txt, summarize_run.py
```

## Run

```
pd/docker/run.sh build
PDK_ROOT=/somewhere/outside/the/repo pd/docker/run.sh install-pdk   # once, sky130 at a pinned open_pdks commit
PDK_ROOT=... pd/docker/run.sh shell
make check                                   # what CI runs
make smoke                                   # toy_counter from the fixture: import runs, synth is a stub and stops it
make run TARGET=<t> HANDOFF=/path/to/package # any target from any package
make import TARGET=<t>                       # one stage
uv run python tools/summarize_run.py runs/<t>/<run_id>
```

The image has uv, shellcheck and yamllint. It has no EDA tools yet. When the stages become real, add OpenROAD, magic and netgen built from pinned source: OSS CAD Suite does not ship them, the conda builds are from 2024, and the prebuilt x86 OpenROAD does not run under Rosetta on Apple silicon.

## Stages

Every stage runs in its own directory under the run, reads only the previous stage's `results/`, and writes `results/DONE.json` with status, wall time, tool versions and the config hash. A stage starts only if the previous status is `ok`. Each stage has a gate; a failed gate is `gate_fail` unless `gates.strict` is false in the config. Stubs report `stub`.

| Stage | Does | Gate |
|---|---|---|
| import | validates the handoff package, copies filelist, RTL and SDC in | package valid |
| synth | RTL to a mapped netlist, honoring dont_use | no unmapped cells, no latches |
| floorplan | die and core, IO pins, macros, taps, power grid | macros legal, core non-empty |
| place | global and detailed placement, timing and congestion driven | no overlaps |
| cts | clock tree with the cts buffers and NDR, post-CTS repair | setup WNS reported, all sinks |
| route | power and special nets, NDR nets, signal nets, fillers | zero router DRC |
| signoff | multi-corner STA, DRC, LVS, antenna | WNS >= 0, DRC and LVS clean |
| export | GDS, DEF, netlist, SDC, SPEF, reports, manifest with checksums | all files present |

Each `flow/stages/<stage>/run.sh` header says the same in one line each.

## Config

`flow/config.py` merges `tech/<node>/platform.yaml` and `dont_use.txt`, then `designs/<target>/design.yaml`, then `RUN_OVERRIDES=<yaml>`. Later wins. The result is `run_config.json` in the run dir, with the layer each key came from, and a flat `run_config.tcl` for Tcl stages. Stage scripts read only those. `${PDK_ROOT}` is expanded at resolve time and must be set.

## Handoff

Import consumes a package as defined by `rtl/handoff` on the rtl branch, schema version 1. It refuses an unknown schema version, `dirty: true`, `lint` not `pass`, a checksum mismatch, and a macro RTL declares that the design does not list. `tests/fixtures/handoff_example/` is a real export of `toy_counter` (see `tests/fixtures/FIXTURE.md`); `test_import.py` asserts the version and one rejection each. Never edit a package in place: PD-side constraint fixes go in `designs/<target>/design.yaml`, RTL-intent fixes go back to rtl as a PR.

## Public and private

| In this repo | Never |
|---|---|
| stage structure, scripts, config schemas | foundry or vendor PDK data of any kind |
| sky130 config pointing at an installed open PDK | EDA scripts that embed foundry rule values |
| design constraints with no foundry values | license servers, foundry material under agreement, internal training material |
| small text QoR summaries | GDS, DEF, SPEF from private-node runs |

Private content is mounted: `PDK_ROOT_PRIVATE` read only, plus an overlay directory with the layout of `tech/sky130/` passed as `TECH=private PRIVATE_TECH_DIR=/path`. `tools/check_no_private.sh` fails CI on anything in `tech/private/` but its two files, on `.lib .db .lef .tf .itf .gds .oas .spef .drc .lvs .cdl` under `pd/`, and on the words in `tools/private_denylist.txt`. If something private is committed, do not delete it in a new commit: tell a lead so history is rewritten and the secret rotated.

## CI

`pd-checks.yml` builds the image with a layer cache and runs `tools/check.sh`: shellcheck, yamllint, ruff, config resolve, pytest, private guard. It then runs import on the fixture and uploads the run summary. The smoke past import waits on real stage scripts. No secrets.

## Lab

Nathan Mara's physical design work lives on the ECE servers at `/misc/scratch/gcNPU/` (world readable, Cadence Genus and Innovus, not open tools) with the PDK at `/misc/scratch/sky130/sky130A`. `pd-flow/<worktree>/{rtl,synthesis,floorplan,pnr,signoff,docs}` is one git worktree per chip variant; `synthesis/README.md`, `floorplan/README.md` and `docs/` hold the measured numbers and the reasoning. What was taken from there: the frame's die, core margin and pin count in `designs/reef_top_sky130_openframe/design.yaml`; the PDK commit ChipFoundry pins, the PDK's dont-use list, tap distance, ring and strap sizes in `tech/sky130/platform.yaml`. Nothing else has run on the full chip yet there either: synthesis outputs, floorplan outputs, `pnr/` and `signoff/` are empty as of 2026-10-15.

## Branches

Branch `pd/<topic>` off `main`. PR with the repo template, someone else reviews, never merge your own.

## Open

1. Engine: direct Yosys and OpenROAD Tcl per stage (this layout) or wrap OpenROAD-flow-scripts or LibreLane. Wrapping gets more out of the box and ties the directory and config names to theirs; direct keeps the stage structure ours and swaps to a commercial backend by replacing `tools-openroad.mk`.
2. Which chip. Nathan's tree builds upstream CoralNPU at `3b453b38` as the whole SoC with the matrix engine, 63 GPIO, a 16-bit host link and JTAG. Aaron's `rtl/hw-drop0` builds the M3 release's `RvvCoreMiniAxi` behind a 10-pin SPI shell. `designs/reef_top_sky130_openframe` carries only what both share: the frame and the SRAM22 macros.
3. Tool source for the real stages, see Run. The lab's Cadence flow is the first private backend candidate.
4. `rcx_rules` in `tech/sky130/platform.yaml` is TBD; sky130A has no extraction tech file.
