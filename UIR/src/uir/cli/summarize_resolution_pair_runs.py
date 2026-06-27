from __future__ import annotations

import argparse
from pathlib import Path

from uir.analysis.resolution_pair_runs import (
    collect_resolution_pair_rows,
    render_resolution_pair_lines,
    write_resolution_pair_outputs,
)
from uir.reporting.aggregate_plots import plot_resolution_pair_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate physical-resolution crop registration summaries.")
    parser.add_argument("runs_root", type=Path, help="UIR/runs directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_resolution_pair_rows(args.runs_root)
    csv_path, json_path = write_resolution_pair_outputs(args.runs_root, rows)
    summary_plot = args.runs_root / "resolution_pair" / "summary" / "resolution_pair_summary.png"
    plot_resolution_pair_summary(rows, summary_plot)
    for line in render_resolution_pair_lines(rows):
        print(line)
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Plot: {summary_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
