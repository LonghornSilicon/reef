# DMA

The current plan contains one DMA channel for linked-list-driven transfers between the SoC memory domains.

The DMA fetches descriptors, generates addresses, and moves data between Coral Core TCMs and external HyperRAM. The memory/interconnect snapshot describes one outstanding transaction and no interrupt in the current implementation reference. These are planning inputs rather than final product guarantees.

## Internal decomposition

The reference layout is preserved as seven authored blocks:

- A left-hand transfer column places **AXI Master · DRAM**, **Transfer FIFO**, and **AXI Master · Core** in order. The DRAM master exchanges memory data with the FIFO; the FIFO exchanges TCM data with the Core master.
- **Control FSM** sequences descriptor fetch, transfer beats, completion, and error handling. It controls the DRAM path and advances the **Address Generator**.
- **Address Generator** produces the source and destination address for each beat and drives the Core-path address.
- **Configuration Registers** hold software-visible configuration and status. They provide start/status to the Control FSM and address parameters to the Address Generator.
- **TL-UL Port** is the configuration boundary; its register-access route terminates at Configuration Registers.

At the DMA boundary, AXI connects toward external HyperRAM and Coral Core TCM access, while TL-UL connects configuration and control to the SoC crossbar. These blocks and routes are architectural responsibilities derived from `DMA_ref.png`; their final RTL module boundaries are not yet confirmed.

## Open decisions

- Descriptor format and ownership
- Address alignment and transfer-size rules
- Error and timeout behavior
- Arbitration against core and host traffic
- Interrupt or polling completion model
- Confirmed external RTL repository and declaration anchor
