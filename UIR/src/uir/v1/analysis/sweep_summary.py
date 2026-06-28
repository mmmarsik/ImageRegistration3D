from __future__ import annotations

import csv
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def collect_sweep_rows(runs_root: Path) -> list[dict]:
    summary_paths = sorted(runs_root.glob("roi*_var*/plots/summary.json"))
    if not summary_paths:
        raise RuntimeError(f"No per-run summary.json files found under {runs_root}")

    rows = [load_summary(path) for path in summary_paths]
    rows.sort(
        key=lambda row: (
            int((row.get("roi_size_xyz") or [0])[0]),
            float(row.get("requested_variance", 0.0)),
        )
    )
    return rows


def write_sweep_outputs(runs_root: Path, rows: list[dict]) -> tuple[Path, Path]:
    columns = [
        "requested_variance",
        "match_count",
        "noise_std",
        "expected_noise_std",
        "noise_std_abs_error",
        "noise_variance_observed",
        "linear_rms_error",
        "linear_mean_abs_error",
        "linear_max_abs_error",
        "translation_l2_error_voxels",
        "translation_mean_abs_error_voxels",
        "translation_max_abs_error_voxels",
        "max_abs_transform_element_error",
        "linear_det_abs_error",
        "expected_transform_path",
        "estimated_transform_path",
        "transform_diff_path",
        "matches_path",
        "run_dir",
    ]

    out_dir = runs_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "variance_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})

    json_path = out_dir / "variance_sweep.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path


def render_sweep_lines(rows: list[dict]) -> list[str]:
    lines = ["Variance sweep summary:"]
    for row in rows:
        variance = format_value(row.get("requested_variance", ""))
        matches = format_value(row.get("match_count", ""))
        noise_std = format_value(row.get("noise_std", ""))
        linear_rms = format_value(row.get("linear_rms_error", ""))
        translation_error = format_value(row.get("translation_l2_error_voxels", ""))
        lines.append(
            f"var={variance:>6}  matches={matches:>4}  "
            f"noise_std={noise_std:>10}  "
            f"linear_rms={linear_rms:>10}  "
            f"translation_l2_voxels={translation_error:>10}"
        )
    return lines
