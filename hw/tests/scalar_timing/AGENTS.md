# Instructions for coding agents: re-run and verify the scalar timing suite

Goal: independently reproduce the scalar-core timing results in this folder and report whether they hold.
You are checking someone else's work. Do not fix it to make it pass. `README.md` explains the suite;
this file is the procedure and the pass criteria.

## Rules

- Do **not** edit expectations, predictions or calibration in `bench/scalar_timing_test.py`, or anything in
  `deliverable/`, `checklist.md` or `notes/`, to make a check pass. A mismatch is a finding: report it.
- Do **not** modify the CoralNPU RTL or check out another commit. Use tag `M3-2026-04-27` (`72d700e`) unmodified.
  `bench/sync.sh` writes only to `$CORALNPU/tests/cocotb/scalar_timing/`.
- Never commit VCDs (25-55 MB each), Bazel output or `results/` (regenerated). Keep VCDs under `$WAVES`, outside the repo.
- Always run tests with `--nocache_test_results`. A cached pass proves nothing.

## 1. Setup

```bash
sudo apt install -y clang zip liblz4-dev          # zip is required: without it every bench reports FAILED
# Bazelisk: https://github.com/bazelbuild/bazelisk (installs Bazel 7.4.1 from CoralNPU's .bazelversion)
git clone https://github.com/google-coral/coralnpu.git && git -C coralnpu checkout M3-2026-04-27
export CORALNPU=$PWD/coralnpu CC=/usr/bin/clang CXX=/usr/bin/clang++ WAVES=$PWD/scalar-timing-waves
mkdir -p "$WAVES"
```

The first build takes about 15 minutes and needs several GB of RAM. If a test is OOM-killed, add
`test --local_test_jobs=4` to `$CORALNPU/.bazelrc.user` and retry.

## 2. Run the suite (from `hw/tests/scalar_timing/`)

```bash
bash bench/sync.sh && (cd "$CORALNPU" && bazel test --nocache_test_results //tests/cocotb/scalar_timing:all)
```

**Pass:** the summary says `Executed 17 out of 17 tests: 17 tests pass.`
On a failure, report the target name and the `measured / expected` table from
`$CORALNPU/bazel-testlogs/tests/cocotb/scalar_timing/<target>/test.log`.

## 3. Reproduce the deliverable

```bash
bash bench/waves.sh trace
python3 bench/collect.py
git diff --stat -- deliverable/timing.json
```

**Pass:** `git diff` shows no change to `deliverable/timing.json`. The model is deterministic, so every cycle
number should come out identical. If it differs, report the classes and fields that changed
(`git diff deliverable/timing.json`), and do not commit the regenerated file.

## 4. Cross-check against the source

For each `results/<bench>.md` written by `collect.py`, compare `measured` with `source-only` (the prediction from
the RTL). Every row where they differ must carry a `sim-calibrated: Dn` note, and `notes/discrepancies.md` must
explain that Dn. **Report** any row that differs from the source without such a note.

## 5. Spot-check the waveform claims

Generate the VCDs, then check each claim with `vcdpeek` (one row per cycle; 1 cycle = 1.25 ns):

```bash
bash bench/waves.sh dep_chain_lw b7_rob
S=core.score
python3 bench/vcdpeek.py cycles "$WAVES/dep_chain_lw.vcd" 900 925 $S.lsu.io_dbus_valid $S.lsu.readFired_valid $S.lsu.io_rd_valid
python3 bench/vcdpeek.py cycles "$WAVES/trace.vcd" 1567.5 1581.25 $S.dispatch.io_inst_0_ready $S.dispatch.io_inst_0_bits_addr $S.dispatch.io_inst_1_ready $S.dispatch.io_inst_1_bits_addr
python3 bench/vcdpeek.py cycles "$WAVES/b7_rob.vcd" 1688 1737 $S.retirement_buffer.io_nSpace $S.retirement_buffer.io_nRetired $S.dispatch.io_inst_0_valid $S.dispatch.io_inst_0_ready
python3 bench/vcdpeek.py params "$WAVES/trace.vcd" FmtPipeRegs
```

What each should show (details and more checks: `deliverable/review_packet.md`, R1-R9):
- **R1 (LSU):** `io_dbus_valid` high once every 4 cycles; between pulses `readFired_valid`, then `io_rd_valid`, then an idle cycle.
- **R2 (mispredict):** the branch at `0x694` dispatches in lane 1; the target `0x69c` dispatches in lane 0 four cycles later (D+4); `0x698` never dispatches.
- **R7 (ROB full):** `io_nSpace` reaches 0 while lane 0 is valid but not ready; when the long `div` writes back, `io_nRetired` is 8 and dispatch resumes.
- **D4 (FPU):** the scalar fpnew groups show `FmtPipeRegs = {0,0,0,0,3}` in this Verilator model (FP32 gets 0 pipeline registers). That is the known sim issue, not a failure.

## 6. Report

Post one summary with:
1. The test summary line from step 2 and the Bazel/Verilator versions used.
2. Whether `deliverable/timing.json` reproduced exactly (step 3).
3. Any source-vs-sim difference without an explanation (step 4).
4. The result of each spot check in step 5, with the cycle times you saw.
5. Anything else that looked wrong, with the file, line and command that shows it.

Known and expected: the FPU benches (B5) run at 1 FP op per 2 cycles in this sim; the hardware does 1 per 5 (D4 in
`notes/discrepancies.md`). A fix for the `PipeRegs` line is planned in Reef, after which B5 changes.
