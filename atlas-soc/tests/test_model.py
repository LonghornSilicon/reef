from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_model import load_model, validate  # noqa: E402


class ArchitectureModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = load_model(ROOT / "architecture" / "soc.yaml")

    def test_checked_in_model_is_valid(self) -> None:
        self.assertEqual(validate(self.model, ROOT), [])

    def test_rejects_dangling_connection(self) -> None:
        model = copy.deepcopy(self.model)
        model["connections"][0]["to"] = "missing-block"
        self.assertTrue(any("unknown block 'missing-block'" in error for error in validate(model, ROOT)))

    def test_rejects_containment_cycle(self) -> None:
        model = copy.deepcopy(self.model)
        system = next(block for block in model["blocks"] if block["id"] == "coral-system")
        system["parent"] = "coral-soc"
        self.assertTrue(any("containment cycle" in error for error in validate(model, ROOT)))

    def test_rejects_unknown_owner(self) -> None:
        model = copy.deepcopy(self.model)
        model["blocks"][0]["owner"] = "not-a-person"
        self.assertTrue(any("unknown owner 'not-a-person'" in error for error in validate(model, ROOT)))

    def test_rejects_missing_fixed_layout(self) -> None:
        model = copy.deepcopy(self.model)
        del model["layouts"]["dma"]
        self.assertTrue(any("composite block 'dma' has no authored layout" in error for error in validate(model, ROOT)))

    def test_rejects_unknown_connection_port(self) -> None:
        model = copy.deepcopy(self.model)
        model["connections"][0]["from_port"] = "missing-port"
        self.assertTrue(any("unknown port 'external-hyperram'.'missing-port'" in error for error in validate(model, ROOT)))

    def test_rejects_unlabelled_port(self) -> None:
        model = copy.deepcopy(self.model)
        model["ports"]["coral-core"]["axi"]["label"] = ""
        self.assertTrue(any("expected a visible label" in error for error in validate(model, ROOT)))

    def test_boot_steps_reference_real_blocks(self) -> None:
        block_ids = {block["id"] for block in self.model["blocks"]}
        for behavior in self.model["behaviors"]:
            for step in behavior["steps"]:
                self.assertLessEqual(set(step["blocks"]), block_ids)

    def test_instruction_steps_reference_real_blocks_and_protocols(self) -> None:
        block_ids = {block["id"] for block in self.model["blocks"]}
        protocol_ids = set(self.model["protocols"])
        for flow in self.model["instruction_flows"]:
            for step in flow["steps"]:
                self.assertLessEqual(set(step["blocks"]), block_ids)
                self.assertIn(step["protocol"], protocol_ids)


if __name__ == "__main__":
    unittest.main()
