from __future__ import annotations

import numpy as np


def as_homogeneous_4x4(mat: np.ndarray) -> np.ndarray:
    if mat.shape == (4, 4):
        return mat.astype(np.float64, copy=False)
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = mat
        return out
    raise RuntimeError(f"Expected 3x4 or 4x4 transform, got shape {mat.shape}")
