from __future__ import annotations

import argparse
import math
import tempfile
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
    block_depth: int = 32,
) -> None:
    if variance < 0.0:
        raise ValueError("variance must be non-negative")
    if block_depth <= 0:
        raise ValueError("block_depth must be positive")

    sigma = math.sqrt(variance)
    nii = nib.load(str(input_path))
    shape = tuple(int(v) for v in nii.shape)

    rng = np.random.default_rng(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(prefix="noise_study_", suffix=".uint16", delete=True) as tmp:
        noisy = np.memmap(tmp.name, dtype=np.uint16, mode="w+", shape=shape)

        for z0 in range(0, shape[2], block_depth):
            z1 = min(z0 + block_depth, shape[2])
            clean_block = np.asarray(nii.dataobj[:, :, z0:z1], dtype=np.float32)
            if variance != 0.0:
                clean_block += rng.normal(0.0, sigma, size=clean_block.shape).astype(np.float32)

            noisy[:, :, z0:z1] = np.rint(np.clip(clean_block, clip_min, clip_max)).astype(np.uint16)

        header = nii.header.copy()
        header.set_data_dtype(np.uint16)
        nib.save(nib.Nifti1Image(noisy, nii.affine, header), str(output_path))


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
    parser.add_argument("--block-depth", type=int, default=32, help="Number of z-slices to process at once.")
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
        block_depth=args.block_depth,
    )
    print(f"Saved: {args.output_path}")
    print(f"Variance: {args.variance}")
    print(f"Seed: {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
