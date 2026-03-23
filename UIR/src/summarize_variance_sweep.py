from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-variance registration summaries into CSV and JSON reports."
    )
    parser.add_argument("runs_root", type=Path, help="Root directory containing run subdirectories.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_root = args.runs_root
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

    columns = [
        "requested_variance",
        "match_count",
        "noise_std",
        "expected_noise_std",
        "noise_std_abs_error",
        "noise_variance_observed",
        "matrix_frobenius_error",
        "linear_frobenius_error",
        "translation_l2_error",
        "max_abs_matrix_error",
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

    print("Variance sweep summary:")
    for row in rows:
        variance = format_value(row.get("requested_variance", ""))
        matches = format_value(row.get("match_count", ""))
        noise_std = format_value(row.get("noise_std", ""))
        matrix_error = format_value(row.get("matrix_frobenius_error", ""))
        translation_error = format_value(row.get("translation_l2_error", ""))
        print(
            f"var={variance:>6}  matches={matches:>4}  "
            f"noise_std={noise_std:>10}  "
            f"matrix_frob={matrix_error:>10}  "
            f"translation_l2={translation_error:>10}"
        )

    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
