from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class VolumeGeometry:
    path: str
    shape_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]

    @property
    def extent_xyz(self) -> tuple[float, float, float]:
        return tuple(float(n) * s for n, s in zip(self.shape_xyz, self.spacing_xyz))

    @property
    def max_xyz(self) -> tuple[float, float, float]:
        return tuple(o + e for o, e in zip(self.origin_xyz, self.extent_xyz))

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "shape_xyz": list(self.shape_xyz),
            "spacing_xyz": list(self.spacing_xyz),
            "origin_xyz": list(self.origin_xyz),
            "extent_xyz": list(self.extent_xyz),
            "max_xyz": list(self.max_xyz),
        }


def read_nifti_geometry(path: Path) -> VolumeGeometry:
    img = nib.load(str(path))
    shape = tuple(int(v) for v in img.shape[:3])
    affine = np.asarray(img.affine, dtype=np.float64)
    spacing = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))
    origin = tuple(float(v) for v in affine[:3, 3])
    return VolumeGeometry(str(path), shape, spacing, origin)


def compare_geometries(
    moving: VolumeGeometry,
    fixed: VolumeGeometry,
    *,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    moving_extent = np.asarray(moving.extent_xyz, dtype=np.float64)
    fixed_extent = np.asarray(fixed.extent_xyz, dtype=np.float64)
    extent_delta = fixed_extent - moving_extent

    common_min = np.maximum(
        np.asarray(moving.origin_xyz, dtype=np.float64),
        np.asarray(fixed.origin_xyz, dtype=np.float64),
    )
    common_max = np.minimum(
        np.asarray(moving.max_xyz, dtype=np.float64),
        np.asarray(fixed.max_xyz, dtype=np.float64),
    )
    common_extent = np.maximum(common_max - common_min, 0.0)
    moving_fraction = common_extent / np.maximum(moving_extent, 1e-12)
    fixed_fraction = common_extent / np.maximum(fixed_extent, 1e-12)

    return {
        "moving": moving.as_dict(),
        "fixed": fixed.as_dict(),
        "extent_delta_fixed_minus_moving_xyz": [float(v) for v in extent_delta],
        "same_physical_extent_within_tolerance": bool(
            np.all(np.abs(extent_delta) <= float(tolerance))
        ),
        "common_physical_roi": {
            "min_xyz": [float(v) for v in common_min],
            "max_xyz": [float(v) for v in common_max],
            "extent_xyz": [float(v) for v in common_extent],
            "moving_fraction_xyz": [float(v) for v in moving_fraction],
            "fixed_fraction_xyz": [float(v) for v in fixed_fraction],
        },
    }
