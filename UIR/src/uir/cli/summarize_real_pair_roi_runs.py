from __future__ import annotations

import argparse
from pathlib import Path

from uir.analysis.real_pair_roi_sweep import (
    collect_real_pair_roi_rows,
    render_real_pair_roi_lines,
    write_real_pair_roi_outputs,
)
from uir.reporting.aggregate_plots import plot_real_pair_roi_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate real-pair centered ROI runs relative to the full-volume transform."
    )
    parser.add_argument("runs_root", type=Path, help="UIR/runs or UIR/runs/real_pair directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_real_pair_roi_rows(args.runs_root)
    csv_path, json_path = write_real_pair_roi_outputs(args.runs_root, rows)
    plot_path = csv_path.parent / "real_pair_roi_relative_error.png"
    plot_real_pair_roi_summary(rows, plot_path)

    for line in render_real_pair_roi_lines(rows):
        print(line)
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
