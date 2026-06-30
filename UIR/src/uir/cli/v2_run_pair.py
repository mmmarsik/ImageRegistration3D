from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from uir.v2.experiment import default_regsift3d_binary, run_pair


def _parse_xyz(raw: str) -> float | tuple[float, float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 3:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    raise argparse.ArgumentTypeError("expected scalar spacing or x,y,z")


def _split_extra_args(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m uir.cli.v2_run_pair",
        description="v2: downsample two PNG stacks and register them with regSift3D.",
    )
    parser.add_argument("moving_stack_dir", type=Path)
    parser.add_argument("fixed_stack_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--ratio", type=float, default=3.0)
    parser.add_argument(
        "--high-spacing",
        type=_parse_xyz,
        default=1.0,
        help="High-resolution voxel spacing, scalar or x,y,z. Used for both stacks unless overridden.",
    )
    parser.add_argument(
        "--moving-high-spacing",
        type=_parse_xyz,
        default=None,
        help="High-resolution spacing for moving_stack_dir, scalar or x,y,z.",
    )
    parser.add_argument(
        "--fixed-high-spacing",
        type=_parse_xyz,
        default=None,
        help="High-resolution spacing for fixed_stack_dir, scalar or x,y,z.",
    )
    parser.add_argument(
        "--same-physical-extent",
        action="store_true",
        help=(
            "After downsampling, choose fixed spacing so fixed_low covers the same "
            "physical extent as moving_low. Useful when stacks have different voxel "
            "counts but represent the same field of view."
        ),
    )
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--resample", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parsed_argv, extra_args = _split_extra_args(raw)
    args = parse_args(parsed_argv)

    if args.ratio <= 0:
        raise SystemExit("--ratio must be positive")

    binary = args.binary if args.binary is not None else default_regsift3d_binary()
    if not Path(binary).exists():
        raise SystemExit(
            f"regSift3D binary not found: {binary}\n"
            "Build it (e.g. UIR/build_all.sh sift3d) or pass --binary / set REGSIFT3D_BIN."
        )

    metadata = run_pair(
        moving_stack_dir=args.moving_stack_dir,
        fixed_stack_dir=args.fixed_stack_dir,
        out_dir=args.out_dir,
        ratio=args.ratio,
        binary=binary,
        high_spacing=args.high_spacing,
        moving_high_spacing=args.moving_high_spacing,
        fixed_high_spacing=args.fixed_high_spacing,
        same_physical_extent=args.same_physical_extent,
        resample=args.resample,
        extra_args=extra_args,
    )
    print(json.dumps(metadata, indent=2))
    return int(metadata["reg_exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
