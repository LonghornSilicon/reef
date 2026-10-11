// Reef top-level RTL (hand-written). Use after the vendored core filelist:
//   -f $CORALNPU_RTL/synth.f -f $REEF_RTL/reef.f   (synthesis)
//   -f $CORALNPU_RTL/sim.f   -f $REEF_RTL/reef.f   (simulation)
${REEF_RTL}/reef_rst_sync.sv
${REEF_RTL}/reef_spi2axi.sv
${REEF_RTL}/reef_axi_err_slave.sv
${REEF_RTL}/reef_top.sv
