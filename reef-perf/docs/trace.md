# Instruction timing trace

`reef_perf --trace FILE` records when each instruction reaches each stage and
writes one line per event, sorted by cycle. `tools/trace_converter` turns that
file into a Kanata log that a pipeline viewer (Konata) opens.

## Trace format

Tab-separated, with a header line: `cycle`, `activity`, `seq`, `pc`,
`instruction`. Activities, in the order an instruction passes them:

| Activity | Set from |
| --- | --- |
| `fetch` | `FetchCycle` |
| `dispatch` | `DispatchCycle` |
| `ScalarExec_Buffer`, `LSU_Buffer`, `Vector_Buffer`, `Matrix_Buffer` | `ExecQueuedCycle`; the name is `QueuedLoc` |
| `ScalarExec`, `LSU`, `Vector`, `Matrix` | `IssueCycle`; the name is `DispatchLoc` |
| `result_ready` | `ResultReadyCycle` |
| `complete` | `CompleteCycle` |
| `retire` | `RetireCycle` |

An event whose cycle was never set is left out. The executing unit sets
`QueuedLoc` and `DispatchLoc`; see interfaces.md.

The simulator keeps every instruction until the end of the run when `--trace`
is given, so memory grows with the instruction count. Without `--trace`
nothing is retained.

## Viewing a trace

```sh
build/bin/reef_perf --elf prog.elf --trace prog.tsv
KONATA=/path/to/konata.sh tools/trace_converter/run_visualizer.sh prog.tsv
```

`run_visualizer.sh` writes `prog.log` next to the trace and opens it if
`KONATA` is set. To cut a long run down, pass an inclusive sequence range:

```sh
tools/trace_converter/run_visualizer.sh prog.tsv 0 1
```

The sequence numbers are decimal and 0-based, so `0 1` includes the first two
instructions.

The Kanata stage names are `F` fetch, `D` dispatch, `SCb`/`LSUb`/`Vb`/`Mb`
the unit queues, `SC`/`LSU`/`V`/`M` execution, `RR` result ready and `C`
complete.
