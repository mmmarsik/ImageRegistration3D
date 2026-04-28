from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from uir.analysis.transform_metrics import matrix_error_stats
from uir.transforms.io import read_transform_csv


CSV_COLUMNS = [
    "pair_tag",
    "roi_size",
    "registration_succeeded",
    "match_count",
    "linear_rms_error",
    "linear_mean_abs_error",
    "linear_max_abs_error",
    "translation_l2_error_voxels",
    "translation_mean_abs_error_voxels",
    "translation_max_abs_error_voxels",
    "max_abs_transform_element_error",
    "linear_det_abs_error",
    "run_dir",
]


def _as_homogeneous_4x4(mat: np.ndarray) -> np.ndarray:
    if mat.shape == (4, 4):
        return mat.astype(np.float64, copy=False)
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = mat
        return out
    raise RuntimeError(f"Expected 3x4 or 4x4 transform, got shape {mat.shape}")


def _crop_to_full_translation(start_xyz: list[int]) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = np.asarray(start_xyz, dtype=np.float64)
    return mat


def _load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _real_pair_root(runs_root: Path) -> Path:
    candidate = runs_root / "real_pair"
    if candidate.exists():
        return candidate
    return runs_root


def collect_real_pair_roi_rows(runs_root: Path) -> list[dict[str, object]]:
    real_root = _real_pair_root(runs_root)
    rows: list[dict[str, object]] = []

    for pair_dir in sorted(path for path in real_root.iterdir() if path.is_dir() and path.name != "summary"):
        full_summary_path = pair_dir / "full_volume" / "summary.json"
        if not full_summary_path.exists():
            continue

        full_summary = _load_summary(full_summary_path)
        full_transform = _as_homogeneous_4x4(read_transform_csv(Path(str(full_summary["transform_path"]))))

        for roi_summary_path in sorted(pair_dir.glob("roi*/summary.json")):
            summary = _load_summary(roi_summary_path)
            roi_size_xyz = summary.get("roi_size_xyz")
            if not roi_size_xyz:
                continue

            row: dict[str, object] = {
                "pair_tag": pair_dir.name,
                "roi_size": int(roi_size_xyz[0]),
                "registration_succeeded": bool(summary.get("registration_succeeded")),
                "match_count": int(summary.get("match_count") or 0),
                "linear_rms_error": "",
                "linear_mean_abs_error": "",
                "linear_max_abs_error": "",
                "translation_l2_error_voxels": "",
                "translation_mean_abs_error_voxels": "",
                "translation_max_abs_error_voxels": "",
                "max_abs_transform_element_error": "",
                "linear_det_abs_error": "",
                "run_dir": str(summary.get("run_dir", roi_summary_path.parent)),
            }

            transform_path = Path(str(summary.get("transform_path", "")))
            if (
                row["registration_succeeded"]
                and transform_path.exists()
                and summary.get("before_roi_start_xyz") is not None
                and summary.get("after_roi_start_xyz") is not None
            ):
                roi_transform = _as_homogeneous_4x4(read_transform_csv(transform_path))
                c_before = _crop_to_full_translation(list(summary["before_roi_start_xyz"]))
                c_after = _crop_to_full_translation(list(summary["after_roi_start_xyz"]))
                roi_transform_full = c_before @ roi_transform @ np.linalg.inv(c_after)
                stats = matrix_error_stats(full_transform[:3, :], roi_transform_full[:3, :])
                row.update(
                    {
                        "linear_rms_error": float(stats["linear_rms_error"]),
                        "linear_mean_abs_error": float(stats["linear_mean_abs_error"]),
                        "linear_max_abs_error": float(stats["linear_max_abs_error"]),
                        "translation_l2_error_voxels": float(stats["translation_l2_error_voxels"]),
                        "translation_mean_abs_error_voxels": float(stats["translation_mean_abs_error_voxels"]),
                        "translation_max_abs_error_voxels": float(stats["translation_max_abs_error_voxels"]),
                        "max_abs_transform_element_error": float(stats["max_abs_transform_element_error"]),
                        "linear_det_abs_error": float(stats["linear_det_abs_error"]),
                    }
                )

            rows.append(row)

    rows.sort(key=lambda row: (str(row["pair_tag"]), int(row["roi_size"])))
    if not rows:
        raise RuntimeError(f"No real-pair ROI summary.json files found under {real_root}")
    return rows


def write_real_pair_roi_outputs(runs_root: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    real_root = _real_pair_root(runs_root)
    out_dir = real_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "real_pair_roi_runs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})

    json_path = out_dir / "real_pair_roi_runs.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path


def render_real_pair_roi_lines(rows: list[dict[str, object]]) -> list[str]:
    lines = [f"Real-pair ROI runs: {len(rows)}"]
    for row in rows:
        error = row.get("translation_l2_error_voxels")
        error_text = "" if error in (None, "") else f" relative_error={float(error):.5f}"
        lines.append(
            f"{row['pair_tag']} roi={row['roi_size']} "
            f"success={row['registration_succeeded']} matches={row['match_count']}{error_text}"
        )
    return lines
