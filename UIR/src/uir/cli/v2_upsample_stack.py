from __future__ import annotations

import argparse
from pathlib import Path

from uir.v2.synth import upsample_png_stack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m uir.cli.v2_upsample_stack",
        description="Synthetic enlargement: upsample a PNG stack by an integer factor (stress-test data).",
    )
    parser.add_argument("src_dir", type=Path)
    parser.add_argument("dst_dir", type=Path)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--resample", default="bilinear", choices=["nearest", "bilinear", "bicubic", "lanczos"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    n = upsample_png_stack(args.src_dir, args.dst_dir, args.factor, args.resample)
    print(f"wrote {n} slices to {args.dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
