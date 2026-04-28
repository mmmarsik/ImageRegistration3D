from __future__ import annotations

import argparse
from pathlib import Path

from uir.perturbations.gaussian_noise import render_noisy_observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Gaussian noise to a NIfTI volume and save the result."
    )
    parser.add_argument("input_path", type=Path, help="Input .nii or .nii.gz file.")
    parser.add_argument("output_path", type=Path, help="Output .nii or .nii.gz file.")
    parser.add_argument("--variance", type=float, required=True, help="Noise variance.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--block-depth", type=int, default=32, help="Number of z-slices to process at once.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    observation_stats = render_noisy_observation(
        args.input_path,
        args.output_path,
        variance=args.variance,
        seed=args.seed,
        block_depth=args.block_depth,
    )
    print(f"Saved: {args.output_path}")
    print(f"Variance: {args.variance}")
    print(f"Seed: {args.seed}")
    print(f"Observation min: {observation_stats['observation_min']}")
    print(f"Observation max: {observation_stats['observation_max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
