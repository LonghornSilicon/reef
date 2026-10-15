# One rule per stage; flow/run_stage.sh enforces the stage contract.

CONFIG := $(RUN_DIR)/run_config.json

$(CONFIG):
	mkdir -p $(RUN_DIR)
	uv run python flow/config.py --tech tech/$(TECH) --design designs/$(TARGET)/design.yaml \
	    $(if $(RUN_OVERRIDES),--override $(RUN_OVERRIDES)) --run-id $(RUN_ID) --out $@

define stage_rule
.PHONY: $(1)
$(1): $(CONFIG) $(if $(2),$(RUN_DIR)/$(2)/results/DONE.json)
	bash flow/run_stage.sh $(1) $(RUN_DIR)
$(RUN_DIR)/$(1)/results/DONE.json: $(1)
endef

$(eval $(call stage_rule,00_import,))
00_import: link
$(eval $(call stage_rule,10_synth,00_import))
$(eval $(call stage_rule,20_floorplan,10_synth))
$(eval $(call stage_rule,30_place,20_floorplan))
$(eval $(call stage_rule,40_cts,30_place))
$(eval $(call stage_rule,50_route,40_cts))
$(eval $(call stage_rule,60_signoff,50_route))
$(eval $(call stage_rule,70_export,60_signoff))

.PHONY: import synth floorplan place cts route signoff export
import: 00_import
synth: 10_synth
floorplan: 20_floorplan
place: 30_place
cts: 40_cts
route: 50_route
signoff: 60_signoff
export: 70_export
