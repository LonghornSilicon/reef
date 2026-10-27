"""Run GPT-Neo through the C++ engine's gpt_neo_cli executable.

A model directory holds a Hugging Face config.json, model.safetensors and
tokenizer.json. The engine reads the first two; tell() also uses the
tokenizer to turn text into token IDs and back.
"""

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from tokenizers import Tokenizer

PRECISIONS = ("fp32", "w8a8")


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


def generate(
    prompt: Sequence[int],
    new_tokens: int,
    model_dir: Path,
    executable: Path,
    precision: str = "fp32",
) -> list[int]:
    """Greedily continue a prompt of token IDs through the C++ engine.

    Returns new_tokens IDs, not including the prompt. The engine does not
    stop at an end-of-text token. It also checks the arguments; anything it
    rejects raises RuntimeError with its message.
    """
    completed = subprocess.run(
        [
            str(executable),
            "--model",
            str(model_dir),
            "--precision",
            precision,
            "--new-tokens",
            str(new_tokens),
            *map(str, prompt),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"gpt_neo_cli failed: {completed.stderr.strip()}")
    return list(map(int, completed.stdout.split()))


def tell(
    prompt: str,
    new_tokens: int,
    model_dir: Path,
    executable: Path,
    precision: str = "fp32",
) -> str:
    """Continue a text prompt, cut before the first end-of-text token.

    Generates new_tokens tokens, then drops everything from the config's
    eos_token_id on: past it the model starts an unrelated new text.
    """
    tokenizer_path = model_dir / "tokenizer.json"
    generated = generate(
        encode_text(prompt, tokenizer_path),
        new_tokens,
        model_dir,
        executable,
        precision,
    )
    config = json.loads((model_dir / "config.json").read_text())
    end = config.get("eos_token_id")
    if end in generated:
        generated = generated[: generated.index(end)]
    return decode_tokens(generated, tokenizer_path)
