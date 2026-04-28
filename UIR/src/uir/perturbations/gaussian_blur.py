from __future__ import annotations

from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

from uir.io.nifti_volume import save_nifti_float32
from uir.observation.model import read_observation_model


def render_blurred_observation(
    input_path: Path,
    output_path: Path,
    *,
    sigma_xy: float,
    block_depth: int = 16,
) -> dict[str, float]:
    if sigma_xy < 0.0:
        raise ValueError("sigma_xy must be non-negative")
    if block_depth <= 0:
        raise ValueError("block_depth must be positive")

    nii = nib.load(str(input_path))
    observation_model = read_observation_model(input_path)
    shape = tuple(int(v) for v in nii.shape)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f"{output_path.name}.tmpf32"
    blurred = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=shape)

    try:
        for z0 in range(0, shape[2], block_depth):
            z1 = min(z0 + block_depth, shape[2])
            block = np.asarray(nii.dataobj[:, :, z0:z1], dtype=np.float32)
            out_block = np.empty_like(block, dtype=np.float32)

            for local_z in range(block.shape[2]):
                slice_xy = np.ascontiguousarray(block[:, :, local_z])
                if sigma_xy == 0.0:
                    out_block[:, :, local_z] = slice_xy
                else:
                    out_block[:, :, local_z] = cv2.GaussianBlur(
                        slice_xy,
                        ksize=(0, 0),
                        sigmaX=sigma_xy,
                        sigmaY=sigma_xy,
                        borderType=cv2.BORDER_REPLICATE,
                    )

            blurred[:, :, z0:z1] = observation_model.render(out_block)

        save_nifti_float32(output_path, blurred, nii.affine, observation_model=observation_model)
    finally:
        del blurred
        tmp_path.unlink(missing_ok=True)

    return {
        "observation_min": observation_model.min_value,
        "observation_max": observation_model.max_value,
        "blur_sigma_xy": float(sigma_xy),
    }
