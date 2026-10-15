# Run inside the image (docker/run.sh shell).
#   make smoke                         toy_counter from the fixture package
#   make run TARGET=<t> HANDOFF=<dir>  link a package, run every stage
#   make <stage> TARGET=<t>            import synth floorplan place cts route signoff export
#   make check                         what CI runs
# RUN_ID= reuses a run dir, TECH= picks tech/<node>, RUN_OVERRIDES= is the last config layer.

TARGET ?= toy_counter
TECH ?= sky130
RUN_ID ?= $(shell date -u +%Y%m%d-%H%M%S)-$(shell git rev-parse --short=8 HEAD)
RUN_DIR := runs/$(TARGET)/$(RUN_ID)

include flow/stages.mk
include flow/tools-openroad.mk
include flow/tools-private.mk

.PHONY: smoke run link check clean

smoke:
	$(MAKE) run TARGET=toy_counter HANDOFF=tests/fixtures/handoff_example

run: link export

link:
ifdef HANDOFF
	mkdir -p inputs && ln -sfn "$(abspath $(HANDOFF))" inputs/$(TARGET)
endif
	@test -e inputs/$(TARGET) || { echo "no inputs/$(TARGET): pass HANDOFF=<package dir>"; exit 1; }

check:
	bash tools/check.sh

clean:
	rm -rf runs; find inputs -type l -delete
