# Link study: Reef at flash 100 MB/s, PSRAM 50 MB/s

How close Reef gets to its memory limits at these link speeds, step by step, on
the 52M model (TinyStories-28M) and a 5.8M model. Start with REPORT.md. Every
step, kept or not, with its evidence, is in LOG.md. The frozen setup and rules
are in SETUP.md.

## Contents

| Path | What it is |
| --- | --- |
| `REPORT.md` | The results: hill climb by hardware tier, ceilings, all steps, other workloads, experiments A and B, open items |
| `LOG.md` | One entry per step: change, source, prediction, result, evidence |
| `SETUP.md` | Fixed setup and the rules for keeping a change |
| `*.png` | Figures used by REPORT.md |
| `summary.md`, `open_items.md`, `phase6.md`, `tiers.md` | Report parts; `scripts/make_report.py` joins them into REPORT.md |
| `runs/` | Simulator output of every run (`*.json`) and its log (`*.log`) |
| `scripts/` | Build, run, metrics and report scripts |
| `patches/sim/` | Simulator changes |
| `patches/firmware/` | One patch per firmware tree |

## Rebuilding

The scripts were run from a scratch directory and use its absolute paths;
change the paths at the top of each script before you use them.

- Simulator: each `patches/sim/simN.patch` applies to `vikshar/dma-in-flight`
  (06bcf53) and holds everything up to that build (`git diff` against 06bcf53).
  `sim.patch` takes in `vikshar/dma-channels` (50b5bca). The builds were used as:

  | Patch | Adds | First used by |
  | --- | --- | --- |
  | `sim.patch` | Per-channel DMA, wait for all transfers but the newest | 3.9 |
  | `sim5.patch` | The newest's channel only for that wait | 3.8 |
  | `sim6.patch` | `dma_setup_overlap` | 3.15 |
  | `sim7.patch` | Wait for transfer N (register 0x20) | 3.16 |
  | `sim8.patch` | Weight stream into the matrix engine (`mtx_stream_b`, B = v25) | 3.17 |
  | `sim9.patch` | KV stream (B = v24) | 3.20 |
  | `sim10.patch` | int4 weight window (unpacked to int8 in the stream) | Phase 6 |

  Matrix counters (`num_macs`, `num_busy_cycles`) are in every patch.
- Firmware: each `patches/firmware/NAME.patch` applies to the end-to-end
  firmware at 83f7152 (Joshua Vigel's `e2e-npu-firmware`, PR #57) and is the
  whole tree at that step:

  | Patch | Step |
  | --- | --- |
  | `fw_s1_snapshot` | 3.1 (also 3.2 and 3.3, which are core settings) |
  | `fw` | 3.14 (3.9 and r2) |
  | `fw37`, `fw38`, `fw310` to `fw313` | Rejected steps 3.7, 3.8, 3.10 to 3.13 |
  | `fw316`, `fw317`, `fw318`, `fw320` | 3.16, 3.17, 3.18, 3.20 |
  | `fwr7`, `fwr12` | Retries of 3.7 and 3.12 |
  | `fwc`, `fwc2` | Data-only combined runs |
  | `fw_w4` | 3.20 with the int4 and model-size options (Phase 6) |
  | `fws1`, `fws2`, `fwsz`, `fws3` | S1, S2, Z1, S3 (`fws3` is the final design) |
  | `fws4`, `fws5` | S4 (rejected), S5 (dropped) |
  | `fwA`, `fwA1`, `fwB` | Experiment A, A on the one-DMA design, experiment B |
  | `fwA3` | Experiment A (and A + int4) on the final design |
  | `e2e_no_matrix` | Vector-only firmware (`-DE2E_NO_MATRIX`), for the matrix vs vector comparison |
  | `fw_vec`, `fw_base` | That patch applied; Joshua's firmware unchanged (baseline builds) |

  Build flags per step are in LOG.md and `scripts/build*.sh`.
