from __future__ import annotations

import csv
import json
from pathlib import Path


REQUIRED_SUMMARY_FIELDS = [
    "run_kind",
    "degradation",
    "transform_tag",
    "roi_size_xyz",
    "blur_sigma_xy",
    "awgn_variance",
    "awgn_seed",
    "match_count",
    "linear_rms_error",
    "linear_mean_abs_error",
    "linear_max_abs_error",
    "translation_l2_error_voxels",
    "translation_mean_abs_error_voxels",
    "translation_max_abs_error_voxels",
    "max_abs_transform_element_error",
    "max_abs_transform_element_component",
    "linear_det_abs_error",
    "noise_std",
    "run_dir",
]


CSV_COLUMNS = [
    "transform_id",
    "transform_tag",
    "roi_size",
    "roi_size_xyz",
    "degradation",
    "blur_sigma_xy",
    "awgn_variance",
    "awgn_seed",
    "match_count",
    "linear_rms_error",
    "linear_mean_abs_error",
    "linear_max_abs_error",
    "translation_l2_error_voxels",
    "translation_mean_abs_error_voxels",
    "translation_max_abs_error_voxels",
    "max_abs_transform_element_error",
    "max_abs_transform_element_component",
    "linear_det_abs_error",
    "noise_std",
    "expected_noise_std",
    "run_dir",
]


def load_synthetic_summary(path: Path) -> dict[str, object]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_SUMMARY_FIELDS if field not in summary]
    if missing:
        raise RuntimeError(f"{path} is missing normalized synthetic fields: {', '.join(missing)}")
    if summary["run_kind"] != "synthetic":
        raise RuntimeError(f"{path} has run_kind={summary['run_kind']!r}, expected 'synthetic'")
    return summary


def collect_synthetic_rows(runs_root: Path) -> list[dict[str, object]]:
    summary_paths = sorted(runs_root.glob("*/**/plots/summary.json"))
    if not summary_paths:
        raise RuntimeError(f"No synthetic plots/summary.json files found under {runs_root}")

    summaries = [load_synthetic_summary(path) for path in summary_paths]
    transform_ids = {
        transform_tag: f"T{i + 1}"
        for i, transform_tag in enumerate(sorted({str(summary["transform_tag"]) for summary in summaries}))
    }

    rows: list[dict[str, object]] = []
    for summary in summaries:
        roi_size_xyz = [int(v) for v in summary["roi_size_xyz"]]
        row = {
            "transform_id": transform_ids[str(summary["transform_tag"])],
            "transform_tag": str(summary["transform_tag"]),
            "roi_size": roi_size_xyz[0],
            "roi_size_xyz": "x".join(str(v) for v in roi_size_xyz),
            "degradation": str(summary["degradation"]),
            "blur_sigma_xy": float(summary["blur_sigma_xy"]),
            "awgn_variance": float(summary["awgn_variance"]),
            "awgn_seed": int(summary["awgn_seed"]),
            "match_count": int(summary["match_count"]),
            "linear_rms_error": float(summary["linear_rms_error"]),
            "linear_mean_abs_error": float(summary["linear_mean_abs_error"]),
            "linear_max_abs_error": float(summary["linear_max_abs_error"]),
            "translation_l2_error_voxels": float(summary["translation_l2_error_voxels"]),
            "translation_mean_abs_error_voxels": float(summary["translation_mean_abs_error_voxels"]),
            "translation_max_abs_error_voxels": float(summary["translation_max_abs_error_voxels"]),
            "max_abs_transform_element_error": float(summary["max_abs_transform_element_error"]),
            "max_abs_transform_element_component": str(summary["max_abs_transform_element_component"]),
            "linear_det_abs_error": float(summary["linear_det_abs_error"]),
            "noise_std": float(summary["noise_std"]),
            "expected_noise_std": summary.get("expected_noise_std", ""),
            "run_dir": str(summary["run_dir"]),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            str(row["transform_tag"]),
            int(row["roi_size"]),
            float(row["blur_sigma_xy"]),
            float(row["awgn_variance"]),
        )
    )
    return rows


def write_synthetic_outputs(runs_root: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    out_dir = runs_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "synthetic_runs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})

    json_path = out_dir / "synthetic_runs.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path


def render_synthetic_summary_lines(rows: list[dict[str, object]]) -> list[str]:
    lines = [f"Synthetic runs: {len(rows)}"]
    groups: dict[tuple[object, object, object], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((row["transform_id"], row["roi_size"], row["blur_sigma_xy"]), []).append(row)

    for (transform_id, roi_size, blur_sigma_xy), group_rows in sorted(groups.items()):
        match_counts = [int(row["match_count"]) for row in group_rows]
        translation_errors = [float(row["translation_l2_error_voxels"]) for row in group_rows]
        lines.append(
            f"{transform_id} roi={roi_size} blur={float(blur_sigma_xy):g}: "
            f"matches {min(match_counts)}..{max(match_counts)}, "
            f"translation_l2_voxels {min(translation_errors):.5f}..{max(translation_errors):.5f}"
        )
    return lines
