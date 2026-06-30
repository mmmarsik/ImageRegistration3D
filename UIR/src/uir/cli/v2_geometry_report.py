from __future__ import annotations

import argparse
import json
from pathlib import Path

from uir.v2.geometry import compare_geometries, read_nifti_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m uir.cli.v2_geometry_report",
        description="Report v2 NIfTI shapes, spacing, physical extents, and common physical ROI.",
    )
    parser.add_argument("moving_nifti", type=Path)
    parser.add_argument("fixed_nifti", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = compare_geometries(
        read_nifti_geometry(args.moving_nifti),
        read_nifti_geometry(args.fixed_nifti),
        tolerance=args.tolerance,
    )
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
