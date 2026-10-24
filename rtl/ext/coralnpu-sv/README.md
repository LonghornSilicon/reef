# coralnpu-sv

The SystemVerilog the Reef build synthesizes: CoralNPU `RvvCoreMiniAxi` from release
`M3-2026-04-27`, the same release `ext/coralnpu` pins. Don't edit these files by hand.

| | |
|---|---|
| Release | [`M3-2026-04-27`](https://github.com/google-coral/coralnpu/releases/tag/M3-2026-04-27), commit `72d700ef` |
| Source | release asset `RvvCoreMiniAxi.sv`, sha256 `74d3bc7e3295353fd879f70a513f7c3fde79912010a1f62f08d815fbcd6f4ff4` |
| Layout | the bundle split one file per module; `synth.f` (design) and `sim.f` (+ `verification/`) list them in compile order |
| Defines | `USE_GENERIC TB_SUPPORT VLEN_128 ZVE32F_ON`, plus `SYNTHESIS` for synthesis |

`TB_SUPPORT` is needed for synthesis too: in M3 the scalar `RetirementBuffer` uses the
`uop_pc`/`last_uop_valid` fields it adds, so the core does not elaborate without it.
