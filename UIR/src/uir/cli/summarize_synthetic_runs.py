from __future__ import annotations

import argparse
from pathlib import Path

from uir.analysis.synthetic_runs import (
    collect_synthetic_rows,
    render_synthetic_summary_lines,
    write_synthetic_outputs,
)
from uir.reporting.aggregate_plots import plot_synthetic_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate normalized synthetic run summaries into compact reports."
    )
    parser.add_argument("runs_root", type=Path, help="UIR/runs directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_synthetic_rows(args.runs_root)
    csv_path, json_path = write_synthetic_outputs(args.runs_root, rows)
    out_dir = args.runs_root / "summary"

    error_plot = out_dir / "synthetic_translation_error_vs_awgn_variance.png"
    matches_plot = out_dir / "synthetic_matches_vs_awgn_variance.png"

    plot_synthetic_metric(
        rows,
        error_plot,
        metric="translation_l2_error_voxels",
        ylabel="translation L2 error (voxels)",
        title="Synthetic Translation Error vs AWGN Variance",
    )
    plot_synthetic_metric(
        rows,
        matches_plot,
        metric="match_count",
        ylabel="match_count",
        title="Synthetic Matches vs AWGN Variance",
    )

    for line in render_synthetic_summary_lines(rows):
        print(line)
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Error plot: {error_plot}")
    print(f"Matches plot: {matches_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
