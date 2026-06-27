from __future__ import annotations

import numpy as np


def as_homogeneous_4x4(mat: np.ndarray) -> np.ndarray:
    """Return a transform as a homogeneous 4x4 float64 matrix.

    Accepts a 4x4 matrix (returned as float64) or a 3x4 matrix (padded with a
    homogeneous bottom row). Any other shape is rejected. This is the single
    canonical implementation shared by the analysis and reporting modules.
    """
    if mat.shape == (4, 4):
        return mat.astype(np.float64, copy=False)
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = mat
        return out
    raise RuntimeError(f"Expected 3x4 or 4x4 transform, got shape {mat.shape}")
