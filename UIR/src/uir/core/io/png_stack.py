from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image

from uir.core.observation.model import ObservationModel


@dataclass(frozen=True)
class StackVolume:
    volume_xyz: np.ndarray
    input_shape_yx: tuple[int, int]
    observation_model: ObservationModel
    decoded_dtype: str
    first_file: str
    last_file: str
    roi_start_xyz: tuple[int, int, int] | None


VOLUME_DTYPES: dict[str, np.dtype] = {
    "float32": np.dtype(np.float32),
    "uint8": np.dtype(np.uint8),
    "uint16": np.dtype(np.uint16),
}


def list_png_paths(input_dir: Path) -> list[Path]:
    pngs = sorted(input_dir.glob("*.png"), key=extract_last_number)
    if not pngs:
        raise RuntimeError(f"No PNG files found in directory: {input_dir}")
    return pngs


def extract_last_number(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if match is None:
        raise ValueError(f"Cannot extract numeric suffix from filename: {path.name}")
    return int(match.group(1))


def _grayscale_palette_levels(img: Image.Image, path: Path) -> list[float]:
    if img.palette is None:
        raise RuntimeError(f"Palette PNG is missing palette data: {path}")

    palette = img.getpalette()
    triplets = [tuple(palette[i : i + 3]) for i in range(0, len(palette), 3)]
    if not triplets or not all(r == g == b for r, g, b in triplets):
        raise RuntimeError(f"Palette PNG is not grayscale: {path}")
    return [float(rgb[0]) for rgb in triplets]


def decode_png_as_gray_native(path: Path) -> tuple[np.ndarray, np.dtype]:
    with Image.open(path) as img:
        if img.mode == "P":
            _grayscale_palette_levels(img, path)
            gray = np.asarray(img.convert("L"), dtype=np.uint8)
            return gray, gray.dtype

        if img.mode == "L":
            gray = np.asarray(img, dtype=np.uint8)
            return gray, gray.dtype

        if img.mode.startswith("I;16"):
            gray = np.asarray(img, dtype=np.uint16)
            return gray, gray.dtype

        if img.mode in ("RGB", "RGBA"):
            rgb = np.asarray(img.convert("RGB"))
            if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])):
                raise RuntimeError(f"RGB PNG is not grayscale: {path}")
            gray = rgb[..., 0].astype(np.uint8, copy=False)
            return gray, gray.dtype

    raise RuntimeError(f"Unsupported PNG mode for grayscale decode in file {path}")


def decode_png_as_gray_f32(path: Path) -> tuple[np.ndarray, np.dtype]:
    gray, decoded_dtype = decode_png_as_gray_native(path)
    return gray.astype(np.float32), decoded_dtype


def infer_png_observation_model(path: Path, decoded_dtype: np.dtype) -> ObservationModel:
    if decoded_dtype == np.uint16:
        return ObservationModel(0.0, 65535.0)

    if decoded_dtype != np.uint8:
        raise RuntimeError(f"Unsupported decoded dtype for observation model: {decoded_dtype}")

    with Image.open(path) as img:
        if img.mode == "P":
            gray_levels = _grayscale_palette_levels(img, path)
            return ObservationModel(min(gray_levels), max(gray_levels))
        if img.mode == "L":
            return ObservationModel(0.0, 255.0)
        if img.mode.startswith("I;16"):
            return ObservationModel(0.0, 65535.0)

    raise RuntimeError(f"Unsupported PNG mode {img.mode} in file {path}")


def inspect_png_stack(input_dir: Path) -> tuple[list[Path], tuple[int, int, int], ObservationModel]:
    pngs = list_png_paths(input_dir)

    shape_yx: tuple[int, int] | None = None
    observation_model: ObservationModel | None = None

    for path in pngs:
        gray, decoded_dtype = decode_png_as_gray_f32(path)
        current_shape_yx = (int(gray.shape[0]), int(gray.shape[1]))
        if shape_yx is None:
            shape_yx = current_shape_yx
        elif current_shape_yx != shape_yx:
            raise RuntimeError(f"Slice shape mismatch: expected {shape_yx}, got {current_shape_yx} in file {path}")

        current_model = infer_png_observation_model(path, decoded_dtype)
        if observation_model is None:
            observation_model = current_model
        elif current_model != observation_model:
            raise RuntimeError(
                f"Observation model mismatch in stack {input_dir}: "
                f"{path.name} has [{current_model.min_value}, {current_model.max_value}] "
                f"but previous slices used [{observation_model.min_value}, {observation_model.max_value}]"
            )

    assert shape_yx is not None
    assert observation_model is not None
    shape_xyz = (shape_yx[1], shape_yx[0], len(pngs))
    return pngs, shape_xyz, observation_model


def load_png_stack_volume(
    input_dir: Path,
    *,
    roi_size_xyz: tuple[int, int, int] | None = None,
    roi_start_xyz: tuple[int, int, int] | None = None,
    observation_model_override: ObservationModel | None = None,
    volume_dtype: str | np.dtype | type = np.float32,
) -> StackVolume:
    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist or is not a directory: {input_dir}")

    pngs = list_png_paths(input_dir)
    full_depth = len(pngs)

    z0 = 0
    x0 = y0 = 0
    roi_x = roi_y = roi_z = None
    if roi_size_xyz is not None:
        roi_x, roi_y, roi_z = roi_size_xyz
        if roi_x <= 0 or roi_y <= 0 or roi_z <= 0:
            raise RuntimeError("ROI size values must be positive.")
        if roi_start_xyz is not None:
            x0, y0, z0 = roi_start_xyz
        else:
            if roi_z > full_depth:
                raise RuntimeError(f"ROI depth {roi_z} exceeds number of slices {full_depth}.")
            z0 = (full_depth - roi_z) // 2
        if z0 < 0 or z0 + roi_z > full_depth:
            raise RuntimeError(
                f"ROI Z range [{z0}, {z0 + roi_z}) is outside available slices [0, {full_depth})."
            )
        pngs = pngs[z0 : z0 + roi_z]

    output_dtype = np.dtype(volume_dtype)
    if output_dtype not in VOLUME_DTYPES.values():
        supported = ", ".join(VOLUME_DTYPES)
        raise RuntimeError(f"Unsupported PNG stack volume dtype {output_dtype}; supported: {supported}.")

    volume_zyx: np.ndarray | None = None
    ref_shape: tuple[int, int] | None = None
    decoded_dtype: np.dtype | None = None
    observation_model: ObservationModel | None = None

    for i, path in enumerate(pngs):
        gray, current_dtype = decode_png_as_gray_native(path)
        current_model = infer_png_observation_model(path, current_dtype)
        if ref_shape is None:
            ref_shape = (int(gray.shape[0]), int(gray.shape[1]))
            decoded_dtype = current_dtype
            observation_model = current_model
        elif gray.shape != ref_shape:
            raise RuntimeError(f"Slice shape mismatch: expected {ref_shape}, got {gray.shape} in file {path}")
        elif current_model != observation_model:
            raise RuntimeError(
                f"Observation model mismatch in stack {input_dir}: "
                f"{path.name} has [{current_model.min_value}, {current_model.max_value}] "
                f"but previous slices used [{observation_model.min_value}, {observation_model.max_value}]"
            )

        if roi_size_xyz is not None:
            if roi_x is None or roi_y is None:
                raise RuntimeError("ROI size is incomplete")
            if roi_start_xyz is None and i == 0:
                x0 = (gray.shape[1] - roi_x) // 2
                y0 = (gray.shape[0] - roi_y) // 2
            if x0 < 0 or y0 < 0 or x0 + roi_x > gray.shape[1] or y0 + roi_y > gray.shape[0]:
                raise RuntimeError(
                    f"ROI XY range [{x0}, {x0 + roi_x}) x [{y0}, {y0 + roi_y}) "
                    f"is outside slice shape {gray.shape[::-1]}."
                )
            gray = gray[y0 : y0 + roi_y, x0 : x0 + roi_x]

        if output_dtype == np.dtype(np.uint8) and current_dtype != np.uint8:
            raise RuntimeError(f"Cannot save {current_dtype} PNG data as uint8 without loss: {path}")

        if volume_zyx is None:
            volume_zyx = np.empty((len(pngs), gray.shape[0], gray.shape[1]), dtype=output_dtype)

        volume_zyx[i] = gray.astype(output_dtype, copy=False)

    assert ref_shape is not None
    assert decoded_dtype is not None
    assert observation_model is not None
    assert volume_zyx is not None

    volume_xyz = np.transpose(volume_zyx, (2, 1, 0))

    return StackVolume(
        volume_xyz=volume_xyz,
        input_shape_yx=ref_shape,
        observation_model=observation_model_override or observation_model,
        decoded_dtype=np.dtype(decoded_dtype).name,
        first_file=pngs[0].name,
        last_file=pngs[-1].name,
        roi_start_xyz=(x0, y0, z0) if roi_size_xyz is not None else None,
    )
