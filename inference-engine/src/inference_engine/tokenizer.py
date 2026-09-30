"""Host-side tokenizer TODOs for the inference engine."""

from collections.abc import Sequence
from pathlib import Path


# @ Tokenizer team
# TODO: Load the selected model's tokenizer assets, then match workloads or
# Hugging Face token IDs exactly, including special-token behavior. Ensure the
# IDs fit the C++ TokenIds representation before passing them to the engine.
def encode_text(text: str, tokenizer_path: Path) -> list[int]:
    """Encode text into model token IDs (not implemented yet)."""
    raise NotImplementedError


# @ Tokenizer team
# TODO: Decode known token IDs for the end-to-end smoke test and verify the
# output against the same tokenizer assets used for encode_text.
def decode_tokens(token_ids: Sequence[int], tokenizer_path: Path) -> str:
    """Decode model token IDs into text (not implemented yet)."""
    raise NotImplementedError
