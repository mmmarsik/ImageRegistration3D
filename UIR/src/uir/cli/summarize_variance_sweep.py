from __future__ import annotations

import argparse
from pathlib import Path

from uir.v1.analysis.sweep_summary import collect_sweep_rows, render_sweep_lines, write_sweep_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-variance registration summaries into CSV and JSON reports."
    )
    parser.add_argument("runs_root", type=Path, help="Root directory containing run subdirectories.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_sweep_rows(args.runs_root)
    csv_path, json_path = write_sweep_outputs(args.runs_root, rows)

    for line in render_sweep_lines(rows):
        print(line)

    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
