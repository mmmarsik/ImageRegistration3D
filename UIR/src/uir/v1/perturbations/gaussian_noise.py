from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from uir.core.io.nifti_volume import save_nifti_float32
from uir.core.observation.model import read_observation_model


def render_noisy_observation(
    input_path: Path,
    output_path: Path,
    *,
    variance: float,
    seed: int = 42,
    block_depth: int = 32,
) -> dict[str, float]:
    if variance < 0.0:
        raise ValueError("variance must be non-negative")
    if block_depth <= 0:
        raise ValueError("block_depth must be positive")

    sigma = float(np.sqrt(variance))
    nii = nib.load(str(input_path))
    observation_model = read_observation_model(input_path)
    shape = tuple(int(v) for v in nii.shape)

    rng = np.random.default_rng(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f"{output_path.name}.tmpf32"

    noisy = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=shape)
    try:
        for z0 in range(0, shape[2], block_depth):
            z1 = min(z0 + block_depth, shape[2])
            clean_block = np.asarray(nii.dataobj[:, :, z0:z1], dtype=np.float32)
            if variance != 0.0:
                clean_block = clean_block + rng.normal(0.0, sigma, size=clean_block.shape).astype(np.float32)

            noisy[:, :, z0:z1] = observation_model.render(clean_block)

        save_nifti_float32(output_path, noisy, nii.affine, observation_model=observation_model)
    finally:
        del noisy
        tmp_path.unlink(missing_ok=True)

    return {
        "observation_min": observation_model.min_value,
        "observation_max": observation_model.max_value,
    }
