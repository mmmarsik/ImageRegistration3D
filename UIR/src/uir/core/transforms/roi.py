from __future__ import annotations

import numpy as np


def roi_expected_transform(
    full_inverse_transform: np.ndarray,
    full_shape_xyz: tuple[int, int, int],
    roi_size_xyz: tuple[int, int, int],
) -> np.ndarray:
    sx, sy, sz = full_shape_xyz
    rx, ry, rz = roi_size_xyz
    x0 = (sx - rx) // 2
    y0 = (sy - ry) // 2
    z0 = (sz - rz) // 2

    crop_shift = np.eye(4, dtype=np.float64)
    crop_shift[0, 3] = x0
    crop_shift[1, 3] = y0
    crop_shift[2, 3] = z0

    roi_expected = np.linalg.inv(crop_shift) @ full_inverse_transform @ crop_shift
    return roi_expected[:3, :]
