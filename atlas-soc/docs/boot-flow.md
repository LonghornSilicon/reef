# Boot and first execution

The boot flow is a behavioral view derived from `SoC presentation.pdf`. It records ordering and concurrency that a containment diagram cannot show.

1. Power is applied while reset is held and initial control values are established.
2. HyperRAM and flash begin their power-on sequences.
3. The selected boot path—host-assisted or flash—loads the initial program into ITCM.
4. Control releases Coral Core, which starts executing from ITCM.
5. Software programs DMA transfers into DTCM or HyperRAM.
6. Independent hardware modules continue their boot FSMs concurrently.
7. Software places kernels in ITCM and enters steady-state execution.

## Failure and recovery questions

The available presentation does not define timeouts, invalid-image handling, failed memory initialization, partial DMA completion, or fallback between host and flash boot. These remain explicit open decisions.

