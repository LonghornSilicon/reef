"""Check text to token IDs and back."""

from pathlib import Path

import pytest
from tokenizers import Tokenizer

from inference_engine.tokenizer import decode_tokens, encode_text


@pytest.mark.unit
def test_text_round_trips(tiny_tokenizer: Path) -> None:
    """Decoding the IDs of any text gives back that text, byte for byte."""
    text = "Once upon a time, Lily saw a naïve dragon 🐉.\n\n  The end."
    ids = encode_text(text, tiny_tokenizer)
    assert decode_tokens(ids, tiny_tokenizer) == text


@pytest.mark.unit
def test_special_tokens_survive_both_ways(tiny_tokenizer: Path) -> None:
    """<|endoftext|> in text encodes to one ID and decodes back to itself."""
    ids = encode_text("The end.<|endoftext|>", tiny_tokenizer)
    assert ids[-1] not in encode_text("The end.", tiny_tokenizer)
    assert decode_tokens(ids, tiny_tokenizer) == "The end.<|endoftext|>"


@pytest.mark.integration
def test_tinystories_ids_match_hugging_face(tmp_path: Path) -> None:
    """TinyStories' own tokenizer gives the IDs Hugging Face does."""
    # Downloads once into the Hugging Face cache.
    path = tmp_path / "tokenizer.json"
    Tokenizer.from_pretrained("roneneldan/TinyStories-28M").save(str(path))
    # No beginning-of-text token is added; 50256 is <|endoftext|>.
    assert encode_text("Once upon a time,", path) == [7454, 2402, 257, 640, 11]
    assert decode_tokens([7454, 50256], path) == "Once<|endoftext|>"
