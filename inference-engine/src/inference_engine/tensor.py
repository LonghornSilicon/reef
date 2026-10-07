"""Call the small host C++ tensor executable from Python."""

import subprocess
from collections.abc import Sequence
from pathlib import Path


def matmul(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    executable: Path,
) -> list[list[float]]:
    """Multiply two nonempty matrices through the C++ host executable."""
    if not left or not right or not left[0] or not right[0]:
        raise ValueError("matrices must be nonempty")
    inner = len(left[0])
    columns = len(right[0])
    if any(len(row) != inner for row in left) or any(
        len(row) != columns for row in right
    ):
        raise ValueError("matrix rows must have equal lengths")
    if inner != len(right):
        raise ValueError("inner dimensions must match")

    values = [float(value) for row in (*left, *right) for value in row]
    payload = f"{len(left)} {inner} {columns}\n"
    payload += " ".join(map(repr, values)) + "\n"
    completed = subprocess.run(
        [str(executable)],
        input=payload,
        text=True,
        capture_output=True,
        check=True,
    )
    output = completed.stdout.split()
    if len(output) < 2:
        raise RuntimeError("C++ tensor executable returned no shape")
    rows, output_columns = map(int, output[:2])
    if (rows, output_columns) != (len(left), columns) or len(output) != (
        2 + rows * output_columns
    ):
        raise RuntimeError("C++ tensor executable returned an invalid shape")
    numbers = list(map(float, output[2:]))
    return [
        numbers[index : index + output_columns]
        for index in range(0, len(numbers), output_columns)
    ]
