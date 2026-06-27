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


def save_nifti(
    output_path: Path,
    volume_xyz: np.ndarray,
    affine: np.ndarray,
    *,
    observation_model: ObservationModel,
    dtype: np.dtype | type = np.float32,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dtype = np.dtype(dtype)
    image = nib.Nifti1Image(np.asarray(volume_xyz, dtype=output_dtype), affine)
    image.header.set_data_dtype(output_dtype)
    image.header["cal_min"] = float(observation_model.min_value)
    image.header["cal_max"] = float(observation_model.max_value)
    nib.save(image, str(output_path))
    write_observation_model(output_path, observation_model)


def save_nifti_float32(
    output_path: Path,
    volume_xyz: np.ndarray,
    affine: np.ndarray,
    *,
    observation_model: ObservationModel,
) -> None:
    save_nifti(output_path, volume_xyz, affine, observation_model=observation_model, dtype=np.float32)
