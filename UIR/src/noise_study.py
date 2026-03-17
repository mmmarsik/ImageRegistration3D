from __future__ import annotations

import argparse
import math
from pathlib import Path

import nibabel as nib
import numpy as np


def add_noise_to_nifti(
    input_path: Path,
    output_path: Path,
    *,
    variance: float,
    seed: int = 42,
    clip_min: float = 0.0,
    clip_max: float = 65535.0,
) -> None:
    if variance < 0.0:
        raise ValueError("variance must be non-negative")

    sigma = math.sqrt(variance)
    nii = nib.load(str(input_path))
    volume = np.asarray(nii.get_fdata(dtype=np.float32), dtype=np.float32)

    rng = np.random.default_rng(seed)
    if variance == 0.0:
        noisy = volume.copy()
    else:
        noise = rng.normal(0.0, sigma, size=volume.shape).astype(np.float32)
        noisy = volume + noise

    noisy = np.rint(np.clip(noisy, clip_min, clip_max)).astype(np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(noisy, nii.affine, nii.header), str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Gaussian noise to a NIfTI volume and save the result."
    )
    parser.add_argument("input_path", type=Path, help="Input .nii or .nii.gz file.")
    parser.add_argument("output_path", type=Path, help="Output .nii or .nii.gz file.")
    parser.add_argument("--variance", type=float, required=True, help="Noise variance.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--clip-min", type=float, default=0.0, help="Minimum output value.")
    parser.add_argument("--clip-max", type=float, default=65535.0, help="Maximum output value.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    add_noise_to_nifti(
        args.input_path,
        args.output_path,
        variance=args.variance,
        seed=args.seed,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )
    print(f"Saved: {args.output_path}")
    print(f"Variance: {args.variance}")
    print(f"Seed: {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
