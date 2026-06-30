from __future__ import annotations

import argparse
from pathlib import Path

from uir.core.io.png_stack import decode_png_as_gray_native, list_png_paths
from uir.v2.downsample import downsample_png_stack
from uir.v2.io import save_nifti_v2


def _parse_xyz(raw: str) -> float | tuple[float, float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 3:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    raise argparse.ArgumentTypeError("expected scalar spacing or x,y,z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream-downsample a PNG stack by a ratio and save it as a .nii (v2)."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ratio", type=float, default=3.0)
    parser.add_argument(
        "--high-spacing",
        type=_parse_xyz,
        default=1.0,
        help="High-resolution voxel spacing, scalar or x,y,z.",
    )
    return parser.parse_args()


def input_shape_xyz(input_dir: Path) -> tuple[int, int, int]:
    pngs = list_png_paths(input_dir)
    gray, _ = decode_png_as_gray_native(pngs[0])
    return (int(gray.shape[1]), int(gray.shape[0]), len(pngs))


def main() -> int:
    args = parse_args()
    if args.ratio <= 0:
        raise SystemExit("--ratio must be positive")

    in_xyz = input_shape_xyz(args.input_dir)
    print(f"input  shape xyz: {in_xyz}")

    low = downsample_png_stack(args.input_dir, args.ratio)
    if isinstance(args.high_spacing, int | float):
        spacing = float(args.high_spacing) * args.ratio
    else:
        spacing = tuple(float(v) * args.ratio for v in args.high_spacing)
    save_nifti_v2(args.output, low, spacing=spacing)

    print(f"output shape xyz: {tuple(int(v) for v in low.shape)}")
    print(f"ratio: {args.ratio}  spacing: {spacing}")
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
