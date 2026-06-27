from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uir.io.nifti_volume import make_affine, save_nifti_float32
from uir.io.png_stack import load_png_stack_volume
from uir.transforms.io import write_transform_csv


def centered_crop_start(shape_xyz: tuple[int, int, int], crop_size_xyz: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple((full - crop) // 2 for full, crop in zip(shape_xyz, crop_size_xyz))


def block_average_downsample(volume_xyz: np.ndarray, ratio: int) -> np.ndarray:
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    usable_shape = tuple((size // ratio) * ratio for size in volume_xyz.shape)
    if any(size == 0 for size in usable_shape):
        raise RuntimeError(f"Ratio {ratio} is too large for volume shape {volume_xyz.shape}")

    cropped = volume_xyz[: usable_shape[0], : usable_shape[1], : usable_shape[2]]
    reshaped = cropped.reshape(
        usable_shape[0] // ratio,
        ratio,
        usable_shape[1] // ratio,
        ratio,
        usable_shape[2] // ratio,
        ratio,
    )
    return reshaped.mean(axis=(1, 3, 5), dtype=np.float32)


def expected_high_crop_to_low_whole_transform(
    ratio: int,
    crop_start_xyz: tuple[int, int, int],
) -> np.ndarray:
    transform = np.zeros((3, 4), dtype=np.float64)
    transform[0, 0] = 1.0 / ratio
    transform[1, 1] = 1.0 / ratio
    transform[2, 2] = 1.0 / ratio
    transform[:, 3] = np.asarray(crop_start_xyz, dtype=np.float64) / float(ratio)
    return transform


def make_resolution_pair(
    *,
    source_stack_dir: Path,
    out_dir: Path,
    ratio: int,
    crop_size_xyz: tuple[int, int, int],
    crop_start_xyz: tuple[int, int, int] | None = None,
    high_spacing: float = 1.0,
) -> dict[str, object]:
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    if high_spacing <= 0.0:
        raise ValueError("high_spacing must be positive")

    stack = load_png_stack_volume(source_stack_dir)
    high = stack.volume_xyz
    full_shape_xyz = tuple(int(v) for v in high.shape)

    if crop_start_xyz is None:
        crop_start_xyz = centered_crop_start(full_shape_xyz, crop_size_xyz)

    x0, y0, z0 = crop_start_xyz
    cx, cy, cz = crop_size_xyz
    if x0 < 0 or y0 < 0 or z0 < 0 or x0 + cx > high.shape[0] or y0 + cy > high.shape[1] or z0 + cz > high.shape[2]:
        raise RuntimeError(
            f"Crop [{crop_start_xyz}, {crop_size_xyz}] is outside source shape {full_shape_xyz}."
        )

    low = block_average_downsample(high, ratio)
    crop = np.asarray(high[x0 : x0 + cx, y0 : y0 + cy, z0 : z0 + cz], dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    low_path = out_dir / "low_res_whole.nii"
    high_crop_path = out_dir / "high_res_crop_clean.nii"
    expected_path = out_dir / "expected_high_crop_to_low_whole_transform.csv"
    metadata_path = out_dir / "resolution_pair.json"

    low_spacing = high_spacing * float(ratio)
    save_nifti_float32(low_path, low, make_affine(low_spacing, low_spacing, low_spacing), observation_model=stack.observation_model)
    save_nifti_float32(
        high_crop_path,
        crop,
        make_affine(high_spacing, high_spacing, high_spacing),
        observation_model=stack.observation_model,
    )

    expected = expected_high_crop_to_low_whole_transform(ratio, crop_start_xyz)
    write_transform_csv(expected_path, expected)

    metadata: dict[str, object] = {
        "run_kind": "resolution_pair",
        "source_stack_dir": str(source_stack_dir),
        "ratio": int(ratio),
        "high_spacing": float(high_spacing),
        "low_spacing": float(low_spacing),
        "source_shape_xyz": list(full_shape_xyz),
        "low_res_shape_xyz": [int(v) for v in low.shape],
        "high_crop_shape_xyz": [int(v) for v in crop.shape],
        "high_crop_start_xyz": [int(v) for v in crop_start_xyz],
        "low_res_whole_path": str(low_path),
        "high_res_crop_clean_path": str(high_crop_path),
        "expected_transform_path": str(expected_path),
        "expected_transform_semantics": "high_res_crop_voxel_to_low_res_whole_voxel",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
