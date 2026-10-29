# Coral architecture overview

This Markdown specification is the editable companion to `SoC_arch.png`. The diagram remains useful historical evidence; `architecture/soc.yaml` is the source of truth consumed by the atlas.

## Coral system boundary

The planning boundary contains the Coral SoC and its external HyperRAM device. External memory is deliberately shown outside the chip so package and controller responsibilities remain visible.

## Coral SoC boundary

The SoC combines Coral Core, DMA, the HyperRAM memory path, the shared TL-UL crossbar, boot/control logic, and peripheral devices. Most blocks remain proposals until their interfaces and RTL ownership are assigned.

## Interconnect

Coral Core presents AXI4 master and slave interfaces. Protocol bridges connect AXI-facing blocks to the SoC's TL-UL fabric. The crossbar routes requests to flash, host link, UART, control, and future peripherals. The current diagrams do not yet settle arbitration, address mapping, clock-domain crossings, or all error behavior; those must be added as the design matures.

## Editing policy

Change the YAML model and Markdown specifications in the same pull request whenever an architectural decision affects both. Use stable block IDs because deep links, issue references, and future RTL anchors depend on them.

