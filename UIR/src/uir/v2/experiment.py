from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from uir.registration.regsift3d import RegSift3D
from uir.v2.downsample import downsample_png_stack
from uir.v2.io import SpacingLike, save_nifti_v2, spacing_xyz
from uir.v2.metrics import evaluate_run


def default_regsift3d_binary() -> Path:
    env = os.environ.get("REGSIFT3D_BIN")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "SIFT3D" / "build" / "bin" / "regSift3D"


def run_pair(
    *,
    moving_stack_dir: Path,
    fixed_stack_dir: Path,
    out_dir: Path,
    ratio: float,
    binary: Path,
    high_spacing: SpacingLike = 1.0,
    moving_high_spacing: SpacingLike | None = None,
    fixed_high_spacing: SpacingLike | None = None,
    same_physical_extent: bool = False,
    resample: bool = False,
    extra_args: Sequence[str] = (),
) -> dict[str, object]:
    if ratio <= 0:
        raise ValueError("ratio must be positive")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratio_value = float(ratio)
    default_high_spacing = spacing_xyz(high_spacing, name="high_spacing")
    moving_high_spacing_xyz = spacing_xyz(
        moving_high_spacing if moving_high_spacing is not None else default_high_spacing,
        name="moving_high_spacing",
    )
    fixed_high_spacing_xyz = spacing_xyz(
        fixed_high_spacing if fixed_high_spacing is not None else default_high_spacing,
        name="fixed_high_spacing",
    )
    moving_low_spacing = tuple(v * ratio_value for v in moving_high_spacing_xyz)
    fixed_low_spacing = tuple(v * ratio_value for v in fixed_high_spacing_xyz)

    moving_path = out_dir / "moving_low.nii"
    fixed_path = out_dir / "fixed_low.nii"

    moving_low = downsample_png_stack(Path(moving_stack_dir), ratio)
    moving_shape = [int(v) for v in moving_low.shape]
    save_nifti_v2(moving_path, moving_low, spacing=moving_low_spacing)
    del moving_low

    fixed_low = downsample_png_stack(Path(fixed_stack_dir), ratio)
    fixed_shape = [int(v) for v in fixed_low.shape]
    if same_physical_extent:
        moving_extent = np.asarray(moving_shape, dtype=np.float64) * np.asarray(
            moving_low_spacing,
            dtype=np.float64,
        )
        fixed_low_spacing = tuple(
            float(v) for v in moving_extent / np.asarray(fixed_shape, dtype=np.float64)
        )
        fixed_high_spacing_xyz = tuple(float(v / ratio_value) for v in fixed_low_spacing)
    save_nifti_v2(fixed_path, fixed_low, spacing=fixed_low_spacing)
    del fixed_low

    matches_path = out_dir / "matches.csv"
    transform_path = out_dir / "transform.csv"

    backend = RegSift3D(binary=Path(binary), resample=resample)
    result = backend.register(
        fixed_path,
        moving_path,
        matches_path=matches_path,
        transform_path=transform_path,
        extra_args=extra_args,
    )

    metadata: dict[str, object] = {
        "run_kind": "v2_downsample_pair",
        "ratio": ratio_value,
        "high_spacing": [float(v) for v in default_high_spacing],
        "moving_high_spacing_xyz": [float(v) for v in moving_high_spacing_xyz],
        "fixed_high_spacing_xyz": [float(v) for v in fixed_high_spacing_xyz],
        "moving_low_spacing_xyz": [float(v) for v in moving_low_spacing],
        "fixed_low_spacing_xyz": [float(v) for v in fixed_low_spacing],
        "same_physical_extent": bool(same_physical_extent),
        "moving_stack_dir": str(moving_stack_dir),
        "fixed_stack_dir": str(fixed_stack_dir),
        "reference": "fixed",
        "moving_low_shape_xyz": moving_shape,
        "fixed_low_shape_xyz": fixed_shape,
        "moving_low_path": str(moving_path),
        "fixed_low_path": str(fixed_path),
        "matches_path": str(matches_path),
        "transform_path": str(transform_path),
        "reg_exit_code": int(result.exit_code),
        "binary": str(binary),
    }
    if len(set(default_high_spacing)) == 1:
        metadata["high_spacing_scalar"] = float(default_high_spacing[0])
    if len(set(moving_low_spacing + fixed_low_spacing)) == 1:
        metadata["low_spacing"] = float(moving_low_spacing[0])
    if result.exit_code == 0 and transform_path.exists():
        metadata["metrics"] = evaluate_run(out_dir)
    (out_dir / "result.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
