from __future__ import annotations

import argparse
import json
from pathlib import Path

from uir.v1.analysis.resolution_pair import make_resolution_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a low-resolution whole volume and high-resolution crop from one source stack."
    )
    parser.add_argument("source_stack_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--ratio", type=int, required=True, help="Low/high voxel spacing ratio.")
    parser.add_argument("--crop-size", nargs=3, type=int, metavar=("X", "Y", "Z"), required=True)
    parser.add_argument("--crop-start", nargs=3, type=int, metavar=("X0", "Y0", "Z0"))
    parser.add_argument("--high-spacing", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = make_resolution_pair(
        source_stack_dir=args.source_stack_dir,
        out_dir=args.out_dir,
        ratio=args.ratio,
        crop_size_xyz=tuple(args.crop_size),
        crop_start_xyz=tuple(args.crop_start) if args.crop_start is not None else None,
        high_spacing=args.high_spacing,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
