from __future__ import annotations

from pathlib import Path

import numpy as np


def read_transform_csv(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split(",")])
    return np.array(rows, dtype=np.float64)


def write_transform_csv(path: Path, mat: np.ndarray) -> None:
    path.write_text(
        "\n".join(
            ",".join(f"{float(value):.12f}" for value in row)
            for row in mat
        )
        + "\n",
        encoding="utf-8",
    )
