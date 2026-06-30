from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from uir.core.io.png_stack import decode_png_as_gray_native, list_png_paths


def _as_integer_ratio(ratio: float) -> int | None:
    rounded = round(ratio)
    if math.isclose(ratio, float(rounded), rel_tol=0.0, abs_tol=1e-9):
        return int(rounded)
    return None


def downsample_png_stack(input_dir: Path, ratio: float) -> np.ndarray:
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    integer_ratio = _as_integer_ratio(float(ratio))
    if integer_ratio is None:
        return _downsample_png_stack_fractional(input_dir, float(ratio))
    return _downsample_png_stack_integer(input_dir, integer_ratio)


def _downsample_png_stack_integer(input_dir: Path, ratio: int) -> np.ndarray:
    pngs = list_png_paths(input_dir)
    full_depth = len(pngs)
    usable_depth = (full_depth // ratio) * ratio
    if usable_depth == 0:
        raise RuntimeError(f"Ratio {ratio} is too large for {full_depth} slices")

    inv_block = 1.0 / float(ratio ** 3)

    low_zyx: np.ndarray | None = None
    ref_shape_hw: tuple[int, int] | None = None
    usable_hw: tuple[int, int] | None = None

    for z_low, z0 in enumerate(range(0, usable_depth, ratio)):
        group_sum: np.ndarray | None = None

        for z in range(z0, z0 + ratio):
            gray, _ = decode_png_as_gray_native(pngs[z])
            shape_hw = (int(gray.shape[0]), int(gray.shape[1]))

            if ref_shape_hw is None:
                ref_shape_hw = shape_hw
                uh = (shape_hw[0] // ratio) * ratio
                uw = (shape_hw[1] // ratio) * ratio
                if uh == 0 or uw == 0:
                    raise RuntimeError(
                        f"Ratio {ratio} is too large for slice shape {shape_hw}"
                    )
                usable_hw = (uh, uw)
                low_zyx = np.empty(
                    (usable_depth // ratio, uh // ratio, uw // ratio),
                    dtype=np.float32,
                )
            elif shape_hw != ref_shape_hw:
                raise RuntimeError(
                    f"Slice shape mismatch: expected {ref_shape_hw}, "
                    f"got {shape_hw} in file {pngs[z]}"
                )

            uh, uw = usable_hw
            cropped = gray[:uh, :uw].astype(np.float64)
            group_sum = cropped if group_sum is None else group_sum + cropped

        assert group_sum is not None and low_zyx is not None and usable_hw is not None
        uh, uw = usable_hw
        block_sums = group_sum.reshape(uh // ratio, ratio, uw // ratio, ratio).sum(axis=(1, 3))
        low_zyx[z_low] = (block_sums * inv_block).astype(np.float32)

    assert low_zyx is not None
    return np.transpose(low_zyx, (2, 1, 0))


def _resize_gray_slice(path: Path, out_w: int, out_h: int) -> np.ndarray:
    with Image.open(path) as im:
        gray = im.convert("L")
        resized = gray.resize((out_w, out_h), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32)


def _downsample_png_stack_fractional(input_dir: Path, ratio: float) -> np.ndarray:
    pngs = list_png_paths(input_dir)
    first, _ = decode_png_as_gray_native(pngs[0])
    in_h, in_w = int(first.shape[0]), int(first.shape[1])
    out_w = int(math.floor(in_w / ratio))
    out_h = int(math.floor(in_h / ratio))
    out_d = int(math.floor(len(pngs) / ratio))
    if out_w <= 0 or out_h <= 0 or out_d <= 0:
        raise RuntimeError(f"Ratio {ratio} is too large for stack shape {(in_w, in_h, len(pngs))}")

    low_zyx = np.empty((out_d, out_h, out_w), dtype=np.float32)
    prev_idx = -1
    prev_slice: np.ndarray | None = None
    next_idx = -1
    next_slice: np.ndarray | None = None

    for z_out in range(out_d):
        z_src = min(z_out * ratio, float(len(pngs) - 1))
        z0 = int(math.floor(z_src))
        z1 = min(z0 + 1, len(pngs) - 1)
        alpha = np.float32(z_src - z0)

        if z0 == next_idx:
            prev_idx, prev_slice = next_idx, next_slice
        elif z0 != prev_idx:
            prev_idx = z0
            prev_slice = _resize_gray_slice(pngs[z0], out_w, out_h)

        if z1 == prev_idx:
            next_idx, next_slice = prev_idx, prev_slice
        elif z1 != next_idx:
            next_idx = z1
            next_slice = _resize_gray_slice(pngs[z1], out_w, out_h)

        assert prev_slice is not None and next_slice is not None
        if alpha == 0.0:
            low_zyx[z_out] = prev_slice
        else:
            low_zyx[z_out] = prev_slice * (np.float32(1.0) - alpha) + next_slice * alpha

    return np.transpose(low_zyx, (2, 1, 0))
