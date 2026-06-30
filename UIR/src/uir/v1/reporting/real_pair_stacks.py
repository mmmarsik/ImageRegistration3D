from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from uir.core.transforms.metrics import apply_transform_to_points, read_match_points
from uir.core.transforms.matrix import as_homogeneous_4x4 as _as_homogeneous_4x4


MODEL_CONSISTENT_THRESHOLD_DEFAULT = 5.0
INTENSITY_CUBOID_RADIUS_DEFAULT = 5


def _volume_display_window(img: nib.Nifti1Image, volume_xyz: np.ndarray) -> tuple[float, float]:
    cal_min = float(img.header["cal_min"])
    cal_max = float(img.header["cal_max"])
    if np.isfinite(cal_min) and np.isfinite(cal_max) and cal_max > cal_min:
        return cal_min, cal_max

    finite = volume_xyz[np.isfinite(volume_xyz)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _gray_rgb_slice(slice_xy: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    lo, hi = window
    gray = np.clip((slice_xy.astype(np.float32, copy=False) - lo) / (hi - lo), 0.0, 1.0)
    gray_u8 = np.rint(gray * 255.0).astype(np.uint8)
    return np.repeat(gray_u8.T[:, :, None], 3, axis=2)


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _draw_crosses(
    image_yx_rgb: np.ndarray,
    points_xy: np.ndarray,
    *,
    radius: int = 3,
    color: tuple[int, int, int] = (255, 0, 0),
) -> None:
    height, width = image_yx_rgb.shape[:2]
    for point in points_xy:
        x = int(np.rint(point[0]))
        y = int(np.rint(point[1]))
        if x < 0 or y < 0 or x >= width or y >= height:
            continue

        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        image_yx_rgb[y, x0:x1] = color
        image_yx_rgb[y0:y1, x] = color


def _group_points_by_slice(points_xyz: np.ndarray, depth: int) -> list[np.ndarray]:
    grouped: list[list[np.ndarray]] = [[] for _ in range(depth)]
    for point in points_xyz:
        z = int(np.rint(point[2]))
        if 0 <= z < depth:
            grouped[z].append(point[:2])
    return [
        np.asarray(points, dtype=np.float64).reshape((-1, 2))
        if points
        else np.empty((0, 2), dtype=np.float64)
        for points in grouped
    ]


def _match_residual_l2(
    source_xyz: np.ndarray,
    reference_xyz: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    if source_xyz.size == 0:
        return np.empty((0,), dtype=np.float64)
    predicted_source_xyz = apply_transform_to_points(transform, reference_xyz)
    return np.linalg.norm(source_xyz - predicted_source_xyz, axis=1)


def _model_consistent_mask(
    source_xyz: np.ndarray,
    reference_xyz: np.ndarray,
    transform: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    residual_l2 = _match_residual_l2(source_xyz, reference_xyz, transform)
    return residual_l2 <= threshold, residual_l2


def save_matched_keypoint_stacks(
    *,
    before_nifti: Path,
    after_nifti: Path,
    matches_path: Path,
    transform: np.ndarray | None = None,
    model_consistent_threshold: float = MODEL_CONSISTENT_THRESHOLD_DEFAULT,
    before_out_dir: Path,
    after_out_dir: Path,
) -> dict[str, object]:
    before_img = nib.load(str(before_nifti))
    after_img = nib.load(str(after_nifti))
    before = np.asarray(before_img.get_fdata(dtype=np.float32), dtype=np.float32)
    after = np.asarray(after_img.get_fdata(dtype=np.float32), dtype=np.float32)

    if before.ndim != 3 or after.ndim != 3:
        raise RuntimeError("Expected 3D before/after NIfTI volumes for keypoint stacks.")

    source_xyz, reference_xyz = read_match_points(matches_path)
    if transform is None:
        consistent_mask = np.zeros(source_xyz.shape[0], dtype=bool)
        residual_l2 = np.empty((0,), dtype=np.float64)
    else:
        consistent_mask, residual_l2 = _model_consistent_mask(
            source_xyz,
            reference_xyz,
            transform,
            model_consistent_threshold,
        )

    before_consistent_by_z = _group_points_by_slice(source_xyz[consistent_mask], before.shape[2])
    after_consistent_by_z = _group_points_by_slice(reference_xyz[consistent_mask], after.shape[2])
    before_window = _volume_display_window(before_img, before)
    after_window = _volume_display_window(after_img, after)

    before_out_dir.mkdir(parents=True, exist_ok=True)
    after_out_dir.mkdir(parents=True, exist_ok=True)

    for z in range(before.shape[2]):
        image = _gray_rgb_slice(before[:, :, z], before_window)
        _draw_crosses(image, before_consistent_by_z[z], color=(0, 255, 0))
        _write_png(before_out_dir / f"slice_{z:04d}.png", image)

    for z in range(after.shape[2]):
        image = _gray_rgb_slice(after[:, :, z], after_window)
        _draw_crosses(image, after_consistent_by_z[z], color=(0, 255, 0))
        _write_png(after_out_dir / f"slice_{z:04d}.png", image)

    result: dict[str, object] = {
        "matched_keypoints_before_stack_dir": str(before_out_dir),
        "matched_keypoints_after_stack_dir": str(after_out_dir),
        "matched_keypoints_cross_radius_px": 3,
        "matched_keypoints_green": "model_consistent_matches",
        "matched_keypoints_outliers_rendered": False,
        "model_consistent_match_threshold": float(model_consistent_threshold),
        "model_consistent_match_count": int(np.count_nonzero(consistent_mask)),
        "model_outlier_match_count": int(source_xyz.shape[0] - np.count_nonzero(consistent_mask)),
    }
    if residual_l2.size > 0:
        result.update(
            {
                "model_consistent_match_fraction": float(np.mean(consistent_mask)),
                "model_match_residual_l2_p50": float(np.percentile(residual_l2, 50.0)),
                "model_match_residual_l2_p95": float(np.percentile(residual_l2, 95.0)),
                "model_match_residual_l2_p99": float(np.percentile(residual_l2, 99.0)),
            }
        )
    return result


def _sample_trilinear_with_mask(
    volume_xyz: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    size_x, size_y, size_z = volume_xyz.shape
    out = np.zeros(x.shape, dtype=np.float32)
    valid = (
        (x >= 0.0)
        & (y >= 0.0)
        & (z >= 0.0)
        & (x <= float(size_x - 1))
        & (y <= float(size_y - 1))
        & (z <= float(size_z - 1))
    )
    if not np.any(valid):
        return out, valid

    xv = x[valid]
    yv = y[valid]
    zv = z[valid]

    x0 = np.floor(xv).astype(np.intp)
    y0 = np.floor(yv).astype(np.intp)
    z0 = np.floor(zv).astype(np.intp)
    x1 = np.minimum(x0 + 1, size_x - 1)
    y1 = np.minimum(y0 + 1, size_y - 1)
    z1 = np.minimum(z0 + 1, size_z - 1)

    xd = (xv - x0).astype(np.float32)
    yd = (yv - y0).astype(np.float32)
    zd = (zv - z0).astype(np.float32)

    c000 = volume_xyz[x0, y0, z0]
    c100 = volume_xyz[x1, y0, z0]
    c010 = volume_xyz[x0, y1, z0]
    c110 = volume_xyz[x1, y1, z0]
    c001 = volume_xyz[x0, y0, z1]
    c101 = volume_xyz[x1, y0, z1]
    c011 = volume_xyz[x0, y1, z1]
    c111 = volume_xyz[x1, y1, z1]

    c00 = c000 * (1.0 - xd) + c100 * xd
    c10 = c010 * (1.0 - xd) + c110 * xd
    c01 = c001 * (1.0 - xd) + c101 * xd
    c11 = c011 * (1.0 - xd) + c111 * xd
    c0 = c00 * (1.0 - yd) + c10 * yd
    c1 = c01 * (1.0 - yd) + c11 * yd

    out[valid] = c0 * (1.0 - zd) + c1 * zd
    return out, valid


def _sample_trilinear(volume_xyz: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    values, _mask = _sample_trilinear_with_mask(volume_xyz, x, y, z)
    return values


def _resample_after_slice_to_before_grid_with_mask(
    after_xyz: np.ndarray,
    before_to_after: np.ndarray,
    z_before: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    src_x = (
        before_to_after[0, 0] * x_grid
        + before_to_after[0, 1] * y_grid
        + before_to_after[0, 2] * float(z_before)
        + before_to_after[0, 3]
    )
    src_y = (
        before_to_after[1, 0] * x_grid
        + before_to_after[1, 1] * y_grid
        + before_to_after[1, 2] * float(z_before)
        + before_to_after[1, 3]
    )
    src_z = (
        before_to_after[2, 0] * x_grid
        + before_to_after[2, 1] * y_grid
        + before_to_after[2, 2] * float(z_before)
        + before_to_after[2, 3]
    )
    return _sample_trilinear_with_mask(after_xyz, src_x, src_y, src_z)


def _resample_after_slice_to_before_grid(
    after_xyz: np.ndarray,
    before_to_after: np.ndarray,
    z_before: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    src_x = (
        before_to_after[0, 0] * x_grid
        + before_to_after[0, 1] * y_grid
        + before_to_after[0, 2] * float(z_before)
        + before_to_after[0, 3]
    )
    src_y = (
        before_to_after[1, 0] * x_grid
        + before_to_after[1, 1] * y_grid
        + before_to_after[1, 2] * float(z_before)
        + before_to_after[1, 3]
    )
    src_z = (
        before_to_after[2, 0] * x_grid
        + before_to_after[2, 1] * y_grid
        + before_to_after[2, 2] * float(z_before)
        + before_to_after[2, 3]
    )
    return _sample_trilinear(after_xyz, src_x, src_y, src_z)


def _signed_diff_rgb(diff_xy: np.ndarray, limit: float) -> np.ndarray:
    scaled = np.clip(diff_xy.astype(np.float32, copy=False) / limit, -1.0, 1.0)
    image = np.empty((diff_xy.shape[1], diff_xy.shape[0], 3), dtype=np.uint8)
    positive = np.maximum(scaled, 0.0).T
    negative = np.maximum(-scaled, 0.0).T

    image[..., 0] = np.rint(255.0 * (1.0 - negative)).astype(np.uint8)
    image[..., 1] = np.rint(255.0 * (1.0 - np.maximum(positive, negative))).astype(np.uint8)
    image[..., 2] = np.rint(255.0 * (1.0 - positive)).astype(np.uint8)
    return image


def _collect_cuboid_samples(
    volume_xyz: np.ndarray,
    points_xyz: np.ndarray,
    radius: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    size_x, size_y, size_z = volume_xyz.shape
    for point in points_xyz:
        x = int(np.rint(point[0]))
        y = int(np.rint(point[1]))
        z = int(np.rint(point[2]))
        if x < 0 or y < 0 or z < 0 or x >= size_x or y >= size_y or z >= size_z:
            continue
        x0 = max(0, x - radius)
        x1 = min(size_x, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(size_y, y + radius + 1)
        z0 = max(0, z - radius)
        z1 = min(size_z, z + radius + 1)
        chunks.append(volume_xyz[x0:x1, y0:y1, z0:z1].ravel())
    if not chunks:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(chunks)


def _sample_volume_for_intensity_stats(volume_xyz: np.ndarray) -> np.ndarray:
    step = max(1, max(volume_xyz.shape) // 128)
    return volume_xyz[::step, ::step, ::step].ravel()


def _robust_intensity_stats(samples: np.ndarray) -> dict[str, float]:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return {"p1": 0.0, "p99": 1.0, "median": 0.0, "scale": 1.0, "median_unit": 0.0}
    p1, p99 = np.percentile(finite, [1.0, 99.0])
    median = float(np.median(finite))
    scale = float(p99 - p1)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.max(finite) - np.min(finite))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    median_unit = (median - float(p1)) / scale
    return {"p1": float(p1), "p99": float(p99), "median": median, "scale": scale, "median_unit": float(median_unit)}


def _estimate_intensity_normalization(
    before_xyz: np.ndarray,
    after_xyz: np.ndarray,
    matches_path: Path | None,
    transform: np.ndarray,
    model_consistent_threshold: float,
    cuboid_radius: int,
) -> dict[str, object]:
    source_xyz = np.empty((0, 3), dtype=np.float64)
    reference_xyz = np.empty((0, 3), dtype=np.float64)
    if matches_path is not None and matches_path.exists():
        source_xyz, reference_xyz = read_match_points(matches_path)

    if source_xyz.shape[0] > 0:
        consistent_mask, _ = _model_consistent_mask(
            source_xyz,
            reference_xyz,
            transform,
            model_consistent_threshold,
        )
        before_samples = _collect_cuboid_samples(before_xyz, source_xyz[consistent_mask], cuboid_radius)
        after_samples = _collect_cuboid_samples(after_xyz, reference_xyz[consistent_mask], cuboid_radius)
    else:
        consistent_mask = np.empty((0,), dtype=bool)
        before_samples = np.empty((0,), dtype=np.float32)
        after_samples = np.empty((0,), dtype=np.float32)

    sample_source = "model_consistent_match_cuboids"
    if before_samples.size == 0 or after_samples.size == 0:
        before_samples = _sample_volume_for_intensity_stats(before_xyz)
        after_samples = _sample_volume_for_intensity_stats(after_xyz)
        sample_source = "fallback_sampled_volume"

    before_stats = _robust_intensity_stats(before_samples)
    after_stats = _robust_intensity_stats(after_samples)
    return {
        "before": before_stats,
        "after": after_stats,
        "sample_source": sample_source,
        "cuboid_radius": int(cuboid_radius),
        "sample_count_before": int(before_samples.size),
        "sample_count_after": int(after_samples.size),
        "model_consistent_match_count": int(np.count_nonzero(consistent_mask)),
    }


def _normalized_slice(slice_xy: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    normalized = (slice_xy.astype(np.float32, copy=False) - stats["p1"]) / stats["scale"]
    return np.clip(normalized, 0.0, 1.0) - stats["median_unit"]


def _normalized_signed_diff_slice(
    before_slice_xy: np.ndarray,
    registered_after_xy: np.ndarray,
    normalization: dict[str, object],
) -> np.ndarray:
    before_stats = normalization["before"]
    after_stats = normalization["after"]
    assert isinstance(before_stats, dict)
    assert isinstance(after_stats, dict)
    return _normalized_slice(before_slice_xy, before_stats) - _normalized_slice(registered_after_xy, after_stats)


def _estimate_signed_diff_limit(
    before_xyz: np.ndarray,
    after_xyz: np.ndarray,
    before_to_after: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    normalization: dict[str, object],
) -> tuple[float, float]:
    depth = before_xyz.shape[2]
    z_step = max(1, depth // 32)
    xy_step = max(1, max(before_xyz.shape[:2]) // 128)
    samples: list[np.ndarray] = []

    for z in range(0, depth, z_step):
        registered_after, valid_mask = _resample_after_slice_to_before_grid_with_mask(
            after_xyz, before_to_after, z, x_grid, y_grid
        )
        diff = _normalized_signed_diff_slice(before_xyz[:, :, z], registered_after, normalization)
        # Out-of-bounds after-resampling даёт ложно-большие значения diff (red borders);
        # они не отражают физическую разницу интенсивностей и не должны влиять на limit.
        diff = np.where(valid_mask, diff, 0.0)
        samples.append(np.abs(diff[::xy_step, ::xy_step]).ravel())

    if not samples:
        return 1.0, 0.0

    abs_sample = np.concatenate(samples)
    finite = abs_sample[np.isfinite(abs_sample)]
    if finite.size == 0:
        return 1.0, 0.0

    abs_p99 = float(np.percentile(finite, 99.0))
    limit = abs_p99
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.max(finite))
    return (limit if limit > 0.0 else 1.0), abs_p99


def save_signed_diff_stack(
    *,
    before_nifti: Path,
    after_nifti: Path,
    transform: np.ndarray,
    matches_path: Path | None = None,
    model_consistent_threshold: float = MODEL_CONSISTENT_THRESHOLD_DEFAULT,
    intensity_cuboid_radius: int = INTENSITY_CUBOID_RADIUS_DEFAULT,
    out_dir: Path,
) -> dict[str, object]:
    before_img = nib.load(str(before_nifti))
    after_img = nib.load(str(after_nifti))
    before = np.asarray(before_img.get_fdata(dtype=np.float32), dtype=np.float32)
    after = np.asarray(after_img.get_fdata(dtype=np.float32), dtype=np.float32)

    if before.ndim != 3 or after.ndim != 3:
        raise RuntimeError("Expected 3D before/after NIfTI volumes for signed diff stack.")

    x = np.arange(before.shape[0], dtype=np.float64)
    y = np.arange(before.shape[1], dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    before_to_after = np.linalg.inv(_as_homogeneous_4x4(transform))
    normalization = _estimate_intensity_normalization(
        before,
        after,
        matches_path,
        transform,
        model_consistent_threshold,
        intensity_cuboid_radius,
    )

    limit, abs_p99 = _estimate_signed_diff_limit(before, after, before_to_after, x_grid, y_grid, normalization)

    out_dir.mkdir(parents=True, exist_ok=True)
    for z in range(before.shape[2]):
        registered_after, valid_mask = _resample_after_slice_to_before_grid_with_mask(
            after, before_to_after, z, x_grid, y_grid
        )
        diff = _normalized_signed_diff_slice(before[:, :, z], registered_after, normalization)
        # Out-of-bounds after-resampling (after-объём не покрыл данную координату before)
        # обнуляется до нейтрального (белого) цвета: иначе zero-padding после нормализации
        # давал бы ложную "разницу" значений и визуально красные ободки, не связанные с
        # реальным изменением интенсивностей.
        diff = np.where(valid_mask, diff, 0.0)
        _write_png(out_dir / f"slice_{z:04d}.png", _signed_diff_rgb(diff, limit))

    before_stats = normalization["before"]
    after_stats = normalization["after"]
    assert isinstance(before_stats, dict)
    assert isinstance(after_stats, dict)

    return {
        "signed_diff_stack_dir": str(out_dir),
        "signed_diff_formula": "normalized_before - normalized_registered_after",
        "signed_diff_out_of_bounds_handling": "after_out_of_bounds_pixels_set_to_neutral_zero",
        "signed_diff_intensity_normalization": "clipped_p1_p99_median_centered",
        "signed_diff_intensity_sample_source": normalization["sample_source"],
        "signed_diff_intensity_cuboid_radius": normalization["cuboid_radius"],
        "signed_diff_intensity_sample_count_before": normalization["sample_count_before"],
        "signed_diff_intensity_sample_count_after": normalization["sample_count_after"],
        "signed_diff_intensity_model_consistent_match_count": normalization["model_consistent_match_count"],
        "signed_diff_before_intensity_p1": before_stats["p1"],
        "signed_diff_before_intensity_p99": before_stats["p99"],
        "signed_diff_before_intensity_median": before_stats["median"],
        "signed_diff_before_intensity_scale": before_stats["scale"],
        "signed_diff_before_intensity_median_unit": before_stats["median_unit"],
        "signed_diff_after_intensity_p1": after_stats["p1"],
        "signed_diff_after_intensity_p99": after_stats["p99"],
        "signed_diff_after_intensity_median": after_stats["median"],
        "signed_diff_after_intensity_scale": after_stats["scale"],
        "signed_diff_after_intensity_median_unit": after_stats["median_unit"],
        "signed_diff_colormap": "blue_negative_white_zero_red_positive",
        "signed_diff_abs_p99": float(abs_p99),
        "signed_diff_display_limit_abs": float(limit),
    }
