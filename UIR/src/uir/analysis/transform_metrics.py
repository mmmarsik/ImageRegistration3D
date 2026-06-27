from __future__ import annotations

from pathlib import Path
import re
import csv

import numpy as np

from uir.transforms.matrix import as_homogeneous_4x4 as _as_homogeneous_4x4


def infer_requested_variance(noisy_path: Path) -> float | None:
    match = re.search(r"_var_(\d+)", noisy_path.name)
    if match is None:
        return None
    return float(int(match.group(1)))


def infer_noisy_path(run_dir: Path, roi_size: int = 250) -> Path:
    candidates = sorted(run_dir.glob(f"volume_B_roi{roi_size}_noise_var_*.nii*"))
    if not candidates:
        raise RuntimeError(f"Cannot find noisy ROI NIfTI in {run_dir}")
    return candidates[0]


def count_match_rows(matches_path: Path) -> int:
    if not matches_path.exists():
        return 0
    with matches_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_match_points(matches_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read regSift3D matches as source and reference XYZ point arrays."""
    if not matches_path.exists():
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    rows: list[list[float]] = []
    with matches_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                values = [float(x) for x in line.split(",")]
                if len(values) != 6:
                    raise RuntimeError(f"Expected 6 columns in {matches_path}, got {len(values)}")
                rows.append(values)

    if not rows:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    matches = np.asarray(rows, dtype=np.float64)
    return matches[:, :3], matches[:, 3:]


def apply_transform_to_points(transform: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    mat = _as_homogeneous_4x4(transform)
    if points_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    points_h = np.c_[points_xyz.astype(np.float64, copy=False), np.ones(points_xyz.shape[0])]
    return (points_h @ mat.T)[:, :3]


def match_residual_rows(
    matches_path: Path,
    transform: np.ndarray,
) -> list[dict[str, float]]:
    source_xyz, reference_xyz = read_match_points(matches_path)
    predicted_source_xyz = apply_transform_to_points(transform, reference_xyz)
    residual_xyz = source_xyz - predicted_source_xyz
    residual_l2 = np.linalg.norm(residual_xyz, axis=1)
    raw_delta_xyz = source_xyz - reference_xyz
    raw_l2 = np.linalg.norm(raw_delta_xyz, axis=1)

    rows: list[dict[str, float]] = []
    for idx in range(source_xyz.shape[0]):
        rows.append(
            {
                "match_index": float(idx),
                "source_x": float(source_xyz[idx, 0]),
                "source_y": float(source_xyz[idx, 1]),
                "source_z": float(source_xyz[idx, 2]),
                "reference_x": float(reference_xyz[idx, 0]),
                "reference_y": float(reference_xyz[idx, 1]),
                "reference_z": float(reference_xyz[idx, 2]),
                "predicted_source_x": float(predicted_source_xyz[idx, 0]),
                "predicted_source_y": float(predicted_source_xyz[idx, 1]),
                "predicted_source_z": float(predicted_source_xyz[idx, 2]),
                "residual_x": float(residual_xyz[idx, 0]),
                "residual_y": float(residual_xyz[idx, 1]),
                "residual_z": float(residual_xyz[idx, 2]),
                "residual_l2": float(residual_l2[idx]),
                "raw_delta_l2": float(raw_l2[idx]),
            }
        )
    return rows


def match_residual_stats(matches_path: Path, transform: np.ndarray) -> dict[str, object]:
    source_xyz, reference_xyz = read_match_points(matches_path)
    predicted_source_xyz = apply_transform_to_points(transform, reference_xyz)
    residual_xyz = source_xyz - predicted_source_xyz
    residual_l2 = np.linalg.norm(residual_xyz, axis=1)
    raw_l2 = np.linalg.norm(source_xyz - reference_xyz, axis=1)

    if residual_l2.size == 0:
        return {
            "match_residual_count": 0,
            "match_raw_l2_mean": None,
            "match_residual_l2_mean": None,
            "match_residual_l2_median": None,
            "match_residual_l2_rms": None,
            "match_residual_l2_p95": None,
            "match_residual_l2_max": None,
            "match_residual_xyz_mean": None,
        }

    return {
        "match_residual_count": int(residual_l2.size),
        "match_raw_l2_mean": float(np.mean(raw_l2)),
        "match_raw_l2_median": float(np.median(raw_l2)),
        "match_raw_l2_max": float(np.max(raw_l2)),
        "match_residual_l2_mean": float(np.mean(residual_l2)),
        "match_residual_l2_median": float(np.median(residual_l2)),
        "match_residual_l2_rms": float(np.sqrt(np.mean(residual_l2 * residual_l2))),
        "match_residual_l2_p95": float(np.percentile(residual_l2, 95)),
        "match_residual_l2_max": float(np.max(residual_l2)),
        "match_residual_xyz_mean": [float(v) for v in np.mean(residual_xyz, axis=0)],
    }


def write_match_residuals_csv(path: Path, matches_path: Path, transform: np.ndarray) -> None:
    rows = match_residual_rows(matches_path, transform)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "match_index",
        "source_x",
        "source_y",
        "source_z",
        "reference_x",
        "reference_y",
        "reference_z",
        "predicted_source_x",
        "predicted_source_y",
        "predicted_source_z",
        "residual_x",
        "residual_y",
        "residual_z",
        "residual_l2",
        "raw_delta_l2",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def transform_component_name(row: int, col: int) -> str:
    if col == 3:
        return ("tx", "ty", "tz")[row]
    return f"linear_r{row}c{col}"


def matrix_element_error_rows(expected: np.ndarray, estimated: np.ndarray) -> list[dict[str, object]]:
    diff = estimated - expected
    rows: list[dict[str, object]] = []
    for row in range(diff.shape[0]):
        for col in range(diff.shape[1]):
            value = float(diff[row, col])
            rows.append(
                {
                    "row": row,
                    "col": col,
                    "component": transform_component_name(row, col),
                    "expected": float(expected[row, col]),
                    "estimated": float(estimated[row, col]),
                    "diff": value,
                    "abs_diff": abs(value),
                }
            )
    return rows


def write_matrix_element_errors_csv(path: Path, expected: np.ndarray, estimated: np.ndarray) -> None:
    rows = matrix_element_error_rows(expected, estimated)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["row", "col", "component", "expected", "estimated", "diff", "abs_diff"],
        )
        writer.writeheader()
        writer.writerows(rows)


def matrix_error_stats(expected: np.ndarray, estimated: np.ndarray) -> dict[str, object]:
    diff = estimated - expected
    linear_diff = diff[:, :3]
    translation_diff = diff[:, 3]
    expected_det = float(np.linalg.det(expected[:, :3]))
    estimated_det = float(np.linalg.det(estimated[:, :3]))
    abs_diff = np.abs(diff)
    abs_linear_diff = np.abs(linear_diff)
    abs_translation_diff = np.abs(translation_diff)
    max_abs_index = tuple(int(v) for v in np.unravel_index(np.argmax(abs_diff), abs_diff.shape))
    max_abs_linear_index = tuple(
        int(v) for v in np.unravel_index(np.argmax(abs_linear_diff), abs_linear_diff.shape)
    )
    max_abs_translation_axis = int(np.argmax(abs_translation_diff))

    return {
        "linear_rms_error": float(np.sqrt(np.mean(linear_diff * linear_diff))),
        "linear_mean_abs_error": float(np.mean(abs_linear_diff)),
        "linear_max_abs_error": float(abs_linear_diff[max_abs_linear_index]),
        "linear_max_abs_error_component": transform_component_name(*max_abs_linear_index),
        "translation_l2_error_voxels": float(np.linalg.norm(translation_diff)),
        "translation_mean_abs_error_voxels": float(np.mean(abs_translation_diff)),
        "translation_max_abs_error_voxels": float(abs_translation_diff[max_abs_translation_axis]),
        "translation_max_abs_error_axis": ["tx", "ty", "tz"][max_abs_translation_axis],
        "max_abs_transform_element_error": float(abs_diff[max_abs_index]),
        "max_abs_transform_element_component": transform_component_name(*max_abs_index),
        "translation_error_xyz": [float(v) for v in translation_diff],
        "expected_linear_det": expected_det,
        "estimated_linear_det": estimated_det,
        "linear_det_abs_error": float(abs(estimated_det - expected_det)),
    }
