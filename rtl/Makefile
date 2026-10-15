# Run inside the image (docker/run.sh shell). Targets: lint, check, sim,
# synth-check, export, clean. TARGET=<name> picks targets/<name>.yaml.

TARGET ?= toy_counter
TOP := $(shell sed -n 's/^top: *//p' targets/$(TARGET).yaml)
FILELIST := filelists/$(TARGET).f
# Verilator's make cannot build under a path with spaces. Point this elsewhere
# if your checkout has one. Inside the image it never does.
REEF_BUILD_DIR ?= build
SIM_DIR := $(REEF_BUILD_DIR)/sim/$(TARGET)
SYNTH_DIR := $(REEF_BUILD_DIR)/synth/$(TARGET)

.PHONY: lint check sim synth-check export clean

lint:
	bash tools/lint.sh

check:
	bash tools/check.sh

TB := $(shell find src -name '$(TOP)_tb.sv')

sim:
	mkdir -p $(SIM_DIR)
	verilator --binary --timing -Wall --timescale 1ns/1ps \
	    -Mdir $(SIM_DIR) -o $(TOP)_tb \
	    -f $(FILELIST) $(TB) --top-module $(TOP)_tb
	$(SIM_DIR)/$(TOP)_tb

synth-check:
	mkdir -p $(SYNTH_DIR)
	yosys -q -l $(SYNTH_DIR)/yosys.log -p \
	    "plugin -i slang; read_slang -f $(FILELIST); synth -top $(TOP); check -assert; tee -q -o $(SYNTH_DIR)/stat.txt stat"
	cat $(SYNTH_DIR)/stat.txt

export:
	uv run python tools/export_handoff.py $(TARGET) $(EXPORT_FLAGS)

clean:
	rm -rf $(REEF_BUILD_DIR) out
