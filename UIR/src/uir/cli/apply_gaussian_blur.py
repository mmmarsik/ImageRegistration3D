from __future__ import annotations

import argparse
from pathlib import Path

from uir.perturbations.gaussian_blur import render_blurred_observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply per-slice Gaussian blur to a NIfTI volume and save the result."
    )
    parser.add_argument("input_path", type=Path, help="Input .nii or .nii.gz file.")
    parser.add_argument("output_path", type=Path, help="Output .nii or .nii.gz file.")
    parser.add_argument("--sigma-xy", type=float, required=True, help="Gaussian sigma in XY slice units.")
    parser.add_argument("--block-depth", type=int, default=16, help="Number of z-slices to process at once.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = render_blurred_observation(
        args.input_path,
        args.output_path,
        sigma_xy=args.sigma_xy,
        block_depth=args.block_depth,
    )
    print(f"Saved: {args.output_path}")
    print(f"Sigma XY: {stats['blur_sigma_xy']}")
    print(f"Observation min: {stats['observation_min']}")
    print(f"Observation max: {stats['observation_max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
