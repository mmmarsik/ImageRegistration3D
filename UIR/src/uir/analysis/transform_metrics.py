from __future__ import annotations

from pathlib import Path
import re
import csv

import numpy as np


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
