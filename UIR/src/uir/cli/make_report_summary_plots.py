from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from uir.v1.reporting.aggregate_plots import (
    plot_real2_roi_comparison,
    plot_real_pair_threshold_comparison,
    plot_synthetic_roi_ladder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build extra summary plots for the UIR report.")
    parser.add_argument("runs_root", type=Path, help="UIR/runs directory.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_threshold_rows(runs_root: Path) -> list[dict[str, object]]:
    specs = [
        ("default", runs_root / "real_pair/real3d_pair_before_to_after/full_volume/summary.json"),
        ("peak020", runs_root / "real_pair/real3d_pair_peak020/full_volume/summary.json"),
        ("corner060", runs_root / "real_pair/real3d_pair_corner060/full_volume/summary.json"),
    ]
    rows: list[dict[str, object]] = []
    for label, path in specs:
        summary = read_json(path)
        rows.append(
            {
                "label": label,
                "match_count": summary["match_count"],
                "match_residual_l2_p95": summary["match_residual_l2_p95"],
            }
        )
    return rows


def build_real2_rows(runs_root: Path) -> list[dict[str, object]]:
    specs = [
        ("roi501 default", runs_root / "real_pair/real3d_pair_another_uint8/roi501/summary.json"),
        ("roi501 peak020", runs_root / "real_pair/real3d_pair_another_uint8_peak020/roi501/summary.json"),
        ("roi650 default", runs_root / "real_pair/real3d_pair_another_uint8/roi650/summary.json"),
        ("roi650 peak020", runs_root / "real_pair/real3d_pair_another_uint8_peak020/roi650/summary.json"),
    ]
    rows: list[dict[str, object]] = []
    for label, path in specs:
        summary = read_json(path)
        rows.append(
            {
                "label": label,
                "match_count": summary["match_count"],
                "model_consistent_match_fraction": summary["model_consistent_match_fraction"],
                "match_residual_l2_p95": summary["match_residual_l2_p95"],
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    runs_root = args.runs_root

    real_summary_dir = runs_root / "real_pair/summary"
    synthetic_summary_dir = runs_root / "summary"

    threshold_plot = real_summary_dir / "real_pair_threshold_comparison.png"
    real2_plot = real_summary_dir / "real2_roi501_roi650_comparison.png"
    roi_ladder_plot = synthetic_summary_dir / "synthetic_roi_ladder.png"

    plot_real_pair_threshold_comparison(build_threshold_rows(runs_root), threshold_plot)
    plot_real2_roi_comparison(build_real2_rows(runs_root), real2_plot)
    plot_synthetic_roi_ladder(read_json(synthetic_summary_dir / "synthetic_runs.json"), roi_ladder_plot)

    print(f"Threshold plot: {threshold_plot}")
    print(f"Real2 ROI plot: {real2_plot}")
    print(f"Synthetic ROI ladder: {roi_ladder_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
