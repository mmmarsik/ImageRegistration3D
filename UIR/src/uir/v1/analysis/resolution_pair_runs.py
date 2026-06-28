from __future__ import annotations

import csv
import json
from pathlib import Path


CSV_COLUMNS = [
    "case_tag",
    "mode",
    "ratio",
    "crop_size",
    "low_res_shape_xyz",
    "high_crop_shape_xyz",
    "registration_succeeded",
    "reg_exit_code",
    "match_count",
    "translation_l2_error_voxels",
    "translation_l2_error_physical",
    "linear_rms_error",
    "run_dir",
    "summary_path",
]


def _format_shape(value: object) -> str:
    if not isinstance(value, list):
        return ""
    return "x".join(str(int(v)) for v in value)


def _optional_float(value: object) -> float | str:
    if value in (None, ""):
        return ""
    return float(value)


def collect_resolution_pair_rows(runs_root: Path) -> list[dict[str, object]]:
    summary_paths = sorted((runs_root / "resolution_pair").glob("*/**/summary.json"))
    rows: list[dict[str, object]] = []
    for path in summary_paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("run_kind") != "resolution_pair":
            continue
        if summary.get("mode") != "resample":
            continue
        crop_shape = summary.get("high_crop_shape_xyz")
        crop_size = int(crop_shape[0]) if isinstance(crop_shape, list) and crop_shape else ""
        rows.append(
            {
                "case_tag": path.parent.parent.name,
                "mode": str(summary["mode"]),
                "ratio": int(summary["ratio"]),
                "crop_size": crop_size,
                "low_res_shape_xyz": _format_shape(summary.get("low_res_shape_xyz")),
                "high_crop_shape_xyz": _format_shape(summary.get("high_crop_shape_xyz")),
                "registration_succeeded": bool(summary["registration_succeeded"]),
                "reg_exit_code": int(summary["reg_exit_code"]),
                "match_count": int(summary["match_count"]),
                "translation_l2_error_voxels": _optional_float(summary.get("translation_l2_error_voxels")),
                "translation_l2_error_physical": _optional_float(summary.get("translation_l2_error_physical")),
                "linear_rms_error": _optional_float(summary.get("linear_rms_error")),
                "run_dir": str(summary["run_dir"]),
                "summary_path": str(path),
            }
        )
    rows.sort(key=lambda row: (int(row["ratio"]), int(row["crop_size"] or 0), str(row["mode"])))
    return rows


def write_resolution_pair_outputs(runs_root: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    out_dir = runs_root / "resolution_pair" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "resolution_pair_runs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})

    json_path = out_dir / "resolution_pair_runs.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path


def render_resolution_pair_lines(rows: list[dict[str, object]]) -> list[str]:
    lines = [f"Resolution-pair runs: {len(rows)}"]
    for row in rows:
        error = row.get("translation_l2_error_voxels") or "n/a"
        lines.append(
            f"ratio={row['ratio']} crop={row['crop_size']} mode={row['mode']} "
            f"success={row['registration_succeeded']} matches={row['match_count']} "
            f"translation_l2_low_voxels={error}"
        )
    return lines
