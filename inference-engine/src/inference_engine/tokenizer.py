"""Text to token IDs and back with a model's Hugging Face tokenizer.json."""

from collections.abc import Sequence
from pathlib import Path

from tokenizers import Tokenizer


def encode_text(text: str, tokenizer_path: Path) -> list[int]:
    """Encode text into model token IDs, as Hugging Face's tokenizer does.

    Special tokens are added only if the tokenizer's own post-processor adds
    them; GPT-Neo's adds none. Special tokens written in the text, such as
    <|endoftext|>, become their IDs.
    """
    return Tokenizer.from_file(str(tokenizer_path)).encode(text).ids


def decode_tokens(token_ids: Sequence[int], tokenizer_path: Path) -> str:
    """Decode model token IDs into text, keeping special tokens.

    The tokenizers library drops special tokens by default; transformers
    keeps them, so this does too.
    """
    return Tokenizer.from_file(str(tokenizer_path)).decode(
        list(token_ids), skip_special_tokens=False
    )
