"""Build the reference engine, run its tests, and drive it from Python.

Runs the Google Test suite through CTest, GPT-Neo from Python on a tiny
random model, the C++ API documentation build, and lint for the whole
module. coral/tests/test_coral.py checks the kernels against the engine.
"""

import json
import math
import random
import shutil
import struct
import subprocess
from pathlib import Path

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from inference_engine.model import decode_tokens, encode_text, generate, tell

MODULE_ROOT = Path(__file__).resolve().parents[3]

HIDDEN = 8
INNER = 16
LAYERS = 2
POSITIONS = 32
PROMPT = [5, 17, 42, 3, 99]

# Tensor name to (shape, values in row-major order).
Tensors = dict[str, tuple[list[int], list[float]]]


@pytest.fixture(scope="session")
def cpp_build_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Configure and build C++ targets in a temporary directory."""
    build_dir = tmp_path_factory.mktemp("cpp-build")
    subprocess.run(
        ["cmake", "-S", str(MODULE_ROOT), "-B", str(build_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    return build_dir


@pytest.fixture
def executable(cpp_build_dir: Path) -> Path:
    """The engine's generation executable."""
    return cpp_build_dir / "bin" / "gpt_neo_cli"


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small byte-level BPE tokenizer.json, built like GPT-Neo's."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        ["Once upon a time, there was a little girl named Lily."], trainer
    )
    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    tokenizer.save(str(path))
    return path


def write_safetensors(path: Path, tensors: Tensors) -> None:
    """Write float tensors as an F32 safetensors file."""
    header: dict[str, object] = {}
    data = bytearray()
    for name, (shape, values) in tensors.items():
        start = len(data)
        data += struct.pack(f"<{len(values)}f", *values)
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [start, len(data)],
        }
    text = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(text)) + text + bytes(data))


def write_tiny_model(
    model_dir: Path, tokenizer: Path, eos_token_id: int | None = None
) -> None:
    """Write a seeded random GPT-Neo with one global and one local layer."""
    vocab = Tokenizer.from_file(str(tokenizer)).get_vocab_size()
    config: dict[str, object] = {
        "activation_function": "gelu_new",
        "attention_layers": ["global", "local"],
        "hidden_size": HIDDEN,
        "intermediate_size": INNER,
        "layer_norm_epsilon": 1e-5,
        "max_position_embeddings": POSITIONS,
        "num_heads": 2,
        "num_layers": LAYERS,
        "vocab_size": vocab,
        "window_size": 4,
    }
    if eos_token_id is not None:
        config["eos_token_id"] = eos_token_id

    generator = random.Random(0)

    def tensor(
        *shape: int, center: float = 0.0
    ) -> tuple[list[int], list[float]]:
        values = [
            center + generator.uniform(-0.5, 0.5)
            for _ in range(math.prod(shape))
        ]
        return list(shape), values

    tensors: Tensors = {
        "transformer.wte.weight": tensor(vocab, HIDDEN),
        "transformer.wpe.weight": tensor(POSITIONS, HIDDEN),
        "transformer.ln_f.weight": tensor(HIDDEN, center=1.0),
        "transformer.ln_f.bias": tensor(HIDDEN),
    }
    for layer in range(LAYERS):
        base = f"transformer.h.{layer}."
        attention = base + "attn.attention."
        tensors |= {
            base + "ln_1.weight": tensor(HIDDEN, center=1.0),
            base + "ln_1.bias": tensor(HIDDEN),
            attention + "q_proj.weight": tensor(HIDDEN, HIDDEN),
            attention + "k_proj.weight": tensor(HIDDEN, HIDDEN),
            attention + "v_proj.weight": tensor(HIDDEN, HIDDEN),
            attention + "out_proj.weight": tensor(HIDDEN, HIDDEN),
            attention + "out_proj.bias": tensor(HIDDEN),
            base + "ln_2.weight": tensor(HIDDEN, center=1.0),
            base + "ln_2.bias": tensor(HIDDEN),
            base + "mlp.c_fc.weight": tensor(INNER, HIDDEN),
            base + "mlp.c_fc.bias": tensor(INNER),
            base + "mlp.c_proj.weight": tensor(HIDDEN, INNER),
            base + "mlp.c_proj.bias": tensor(HIDDEN),
        }

    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config))
    write_safetensors(model_dir / "model.safetensors", tensors)
    shutil.copy(tokenizer, model_dir / "tokenizer.json")


def test_cpp(cpp_build_dir: Path) -> None:
    """Run every C++ test through CTest; failures show CTest's output."""
    subprocess.run(
        ["ctest", "--test-dir", str(cpp_build_dir), "--output-on-failure"],
        check=True,
    )


@pytest.mark.parametrize("precision", ["fp32", "w8a8"])
def test_generate_returns_new_tokens_from_the_vocabulary(
    tmp_path: Path, tiny_tokenizer: Path, executable: Path, precision: str
) -> None:
    """The engine returns exactly the requested count of known token IDs."""
    write_tiny_model(tmp_path, tiny_tokenizer)
    vocab = json.loads((tmp_path / "config.json").read_text())["vocab_size"]

    generated = generate(PROMPT, 8, tmp_path, executable, precision)
    assert len(generated) == 8
    assert all(0 <= token < vocab for token in generated)
    assert generate(PROMPT, 0, tmp_path, executable, precision) == []


def test_generate_continues_its_own_output(
    tmp_path: Path, tiny_tokenizer: Path, executable: Path
) -> None:
    """Prompting with the first generated tokens yields the rest.

    Greedy decoding guarantees this, so it fails if Python reorders the
    prompt or drops a generated token.
    """
    write_tiny_model(tmp_path, tiny_tokenizer)
    generated = generate(PROMPT, 8, tmp_path, executable)
    rest = generate(PROMPT + generated[:3], 5, tmp_path, executable)
    assert rest == generated[3:]


def test_generate_reports_engine_errors(
    tmp_path: Path, executable: Path
) -> None:
    """Bad arguments and engine failures raise with the engine's message."""
    with pytest.raises(RuntimeError, match="usage"):
        generate([], 1, tmp_path, executable)
    with pytest.raises(RuntimeError, match=r"config\.json"):
        generate(PROMPT, 1, tmp_path, executable)


def test_tell_stops_before_end_of_text(
    tmp_path: Path, tiny_tokenizer: Path, executable: Path
) -> None:
    """tell() decodes the continuation up to the config's eos_token_id."""
    write_tiny_model(tmp_path, tiny_tokenizer)
    tokenizer = tmp_path / "tokenizer.json"
    prompt = "Once upon a time"
    ids = generate(encode_text(prompt, tokenizer), 10, tmp_path, executable)
    # Without an eos_token_id nothing is cut.
    assert tell(prompt, 10, tmp_path, executable) == decode_tokens(
        ids, tokenizer
    )

    # Declare a token that first appears partway through to be end-of-text.
    cut = next(i for i in range(1, len(ids)) if ids[i] not in ids[:i])
    write_tiny_model(tmp_path, tiny_tokenizer, eos_token_id=ids[cut])
    assert tell(prompt, 10, tmp_path, executable) == decode_tokens(
        ids[:cut], tokenizer
    )


@pytest.mark.slow
def test_lint() -> None:
    """Check Python and C++ lint without modifying source files."""
    subprocess.run(
        ["uv", "run", "--locked", "ruff", "check", "."],
        cwd=MODULE_ROOT,
        check=True,
    )
    subprocess.run(
        ["uv", "run", "--locked", "ruff", "format", "--check", "."],
        cwd=MODULE_ROOT,
        check=True,
    )
    subprocess.run(["bash", "tools/check_cpp.sh"], cwd=MODULE_ROOT, check=True)


def test_doxygen_builds_without_warnings(tmp_path: Path) -> None:
    """Generate the C++ API documentation with warnings treated as errors."""
    output_dir = tmp_path / "doxygen"
    config = (MODULE_ROOT / "Doxyfile").read_text()
    config += f'\nOUTPUT_DIRECTORY = "{output_dir}"\n'
    result = subprocess.run(
        ["doxygen", "-"],
        input=config,
        text=True,
        cwd=MODULE_ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert (output_dir / "html" / "index.html").is_file()
