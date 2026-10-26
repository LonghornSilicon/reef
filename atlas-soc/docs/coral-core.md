# Coral Core

Coral Core is a scalar/vector RISC-V compute block with two software-managed tightly coupled memories and uncached AXI access to the surrounding SoC.

## AXI ingress and egress

The core exposes one **AXI Slave** ingress and one **AXI Master** egress. Inbound requests pass through **AXI to Fabric** and **Fabric Mux**, then branch toward the ITCM and DTCM arbitration paths. Outbound uncached requests pass from **Scalar + Vector Core** through **Arbiter** to **AXI Master**. DMA also reaches the Coral Core boundary over AXI for TCM access.

## Fabric, debug, and TCM arbitration

- **Fabric Mux** sends inbound Fabric requests to **DTCM Arbiter** or **ITCM Arbiter**.
- Scalar and vector data reach DTCM over DBUS. Instruction fetch reaches **IBUS Mux**, then ITCM Arbiter, over IBUS.
- **Debug** reaches both TCM arbiters over the CTRL fabric, separate from inbound data Fabric traffic.
- DTCM Arbiter drives the DTCM banks over DBUS; ITCM Arbiter drives the ITCM bank over IBUS.
- Uncached EBUS requests are selected by the outbound Arbiter before reaching AXI Master.

The TCMs are single-port, so arbitration is an architectural behavior rather than an implementation detail. The current source material says the core wins over inbound AXI traffic when both contend for a TCM.

## Tightly coupled memories

The drill-down uses the canonical bank organization from the references:

- **ITCM · 1 Bank** contains one **128×512 Bank**: 8 KiB total, with one 16-byte row read per cycle.
- **DTCM · 4 Banks** contains **128×512 Bank 0** through **Bank 3**: 32 KiB total, with one-cycle read latency.

## Compute hierarchy

**Scalar + Vector Core** is the compute-cluster enclosure. It contains three peers:

- **Scalar Core**: Fetch · 128-bit ×4, Decode ×4, ALU ×4, Branch Unit ×4, Multiply, Divide, Load/Store Unit · LSU, Control/Status Registers · CSR, and Integer Register File · 32-bit. The register-file drill-down exposes `x0 · x1 · … · x31` and its Scoreboard.
- **Vector Execution Engine**: Vector Command Queue, Vector ALU ×2, Vector MAC/MUL/DIV, and Vector Register File · 128-bit. Its register-file drill-down exposes `v0 · v1 · … · v31` and its Scoreboard.
- **16×16 Systolic Array**: a solid black-box accelerator instance commanded by the Scalar Core through `ARRAY CMD`. Its internal implementation is intentionally hidden at this integration level.

The separate **Systolic Array** reference hierarchy remains a standalone 4×4 evidence model. It documents activation, PE-grid, accumulation, and processing-element concepts without turning the integrated 16×16 instance into an expanded container.

## Systolic Array and processing element

The standalone reference model preserves the composition: **Unified Multi-Bank Activation Buffer**, **4×4 PE + Weight Buffer Grid**, **Output Buffers + Accumulators ×4**, and **Processing Element Detail**. Their drill-downs expose four activation banks, 16 row/column cells named `PE R0C0 + WB` through `PE R3C3 + WB`, and four accumulator/output-buffer paths. Activation flows left-to-right across each grid row; partial sums flow top-to-bottom through each column.

Processing Element Detail exposes Input Activation, Weight, Input Partial Sum, Multiply, Add, Activation Register, Output Activation, Output Register, and Output Partial Sum. The activation feeds Multiply and the forwarding register; Weight feeds Multiply; the product and incoming partial sum feed Add; registered activation exits to the right and the registered accumulated sum exits downward.

## Internal routing

Instruction fetches use IBUS to ITCM. Data requests in the DTCM range use DBUS. Other data addresses use EBUS and leave through the AXI master. The internal Fabric carries inbound AXI memory access, while the CTRL fabric carries Debug access to both TCM arbiters.

## Open decisions

Confirmed RTL source anchors, module ownership, final TCM sizes, debug integration, error recovery, and clock/reset-domain details remain to be recorded.
