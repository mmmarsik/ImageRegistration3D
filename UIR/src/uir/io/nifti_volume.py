from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from uir.observation.model import ObservationModel, write_observation_model


def make_affine(sx: float, sy: float, sz: float) -> np.ndarray:
    return np.array(
        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def save_nifti_float32(
    output_path: Path,
    volume_xyz: np.ndarray,
    affine: np.ndarray,
    *,
    observation_model: ObservationModel,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(volume_xyz, dtype=np.float32), affine)
    image.header.set_data_dtype(np.float32)
    image.header["cal_min"] = float(observation_model.min_value)
    image.header["cal_max"] = float(observation_model.max_value)
    nib.save(image, str(output_path))
    write_observation_model(output_path, observation_model)
