from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import nibabel as nib
import numpy as np

from uir.core.io.nifti_volume import make_affine


SpacingLike = float | Sequence[float]


def spacing_xyz(value: SpacingLike, *, name: str = "spacing") -> tuple[float, float, float]:
    if isinstance(value, int | float):
        xyz = (float(value), float(value), float(value))
    else:
        raw = tuple(float(v) for v in value)
        if len(raw) != 3:
            raise ValueError(f"{name} must be a scalar or three xyz values")
        xyz = raw

    if any(v <= 0.0 for v in xyz):
        raise ValueError(f"{name} values must be positive")
    return xyz


def origin_xyz(value: Sequence[float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    raw = tuple(float(v) for v in value)
    if len(raw) != 3:
        raise ValueError("origin must have three xyz values")
    return raw


def save_nifti_v2(
    output_path: Path,
    volume_xyz: np.ndarray,
    spacing: SpacingLike,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> None:
    sx, sy, sz = spacing_xyz(spacing)
    ox, oy, oz = origin_xyz(origin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    affine = make_affine(sx, sy, sz)
    affine[:3, 3] = [ox, oy, oz]
    image = nib.Nifti1Image(np.asarray(volume_xyz, dtype=np.float32), affine)
    image.header.set_data_dtype(np.float32)
    nib.save(image, str(output_path))
