# RTL team Git workflow

The RTL team works on the **`rtl-dev`** branch. It does not change `main`.
`main` only receives RTL through a reviewed `rtl-dev → main` pull request at a milestone
(for example, the RTL drop to PD), so other teams' work on `main` and on their own
branches is never affected by day-to-day RTL changes.

```
main                        ← agreed design only (PR + 1 approval + lint)
  └── rtl-dev               ← RTL team integration branch (PR + 1 approval)
       ├── rtl/<name>/<topic>   e.g. rtl/krish/datapath-audit
       ├── rtl/<name>/<topic>   e.g. rtl/alex/gemm
       └── ...
```

## Day to day

1. Branch from `rtl-dev`: `git switch rtl-dev && git pull && git switch -c rtl/<name>/<topic>`
2. Commit and push your branch as often as you like.
3. Open a PR **into `rtl-dev`** (not `main`). Use a draft PR while it's still in progress.
4. One RTL teammate reviews and approves, then merge.

## Milestones

At each milestone the RTL leads open one PR `rtl-dev → main`. That PR goes through
`main`'s rules (1 approval + the `lint` check).

## Rules

- **Never open RTL PRs against `main` directly.**
- **Don't hand-edit `rtl/vendor/coralnpu/`.** It is generated from CoralNPU release
  `M3-2026-04-27`; see `rtl/vendor/UPSTREAM.md` and `docs/baseline-m3.md`.
- **This repo is public. Never commit foundry/PDK/NDA material** (memory-compiler
  macros, `.lib`/`.lef`/`.gds`, PDK files, anything under a TSMC/imec agreement).
  Those go in a separate private LonghornSilicon repo.
- Don't commit build or sim outputs (`obj_dir/`, `sim_build/`, waveforms, logs); see `.gitignore`.

## Baseline

- CoralNPU release `M3-2026-04-27` (commit `72d700e`), top `RvvCoreMiniAxi`.
- Filelists: `rtl/vendor/coralnpu/synth.f`, `rtl/vendor/coralnpu/sim.f`.
- Regenerate: `scripts/regen_coralnpu.sh`. Lint: `scripts/lint_vendor.sh`.
- Verification record: `results/m3_verification.md`. RTL map: `docs/rtl-map-v0.md`.
