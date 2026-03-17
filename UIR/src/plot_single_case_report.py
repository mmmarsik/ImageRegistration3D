from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def extract_last_number(path: Path) -> int:
    m = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if m is None:
        raise ValueError(f"Cannot extract numeric suffix from filename: {path.name}")
    return int(m.group(1))


def read_transform_csv(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split(",")])
    return np.array(rows, dtype=np.float64)


def stack_shape_xyz(stack_dir: Path) -> tuple[int, int, int]:
    pngs = sorted(stack_dir.glob("*.png"), key=extract_last_number)
    if not pngs:
        raise RuntimeError(f"No PNG files found in {stack_dir}")
    first = cv2.imread(str(pngs[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"Failed to read image: {pngs[0]}")
    if first.ndim == 3:
        first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    return (int(first.shape[1]), int(first.shape[0]), len(pngs))


def roi_expected_transform(
    full_inverse_transform: np.ndarray,
    full_shape_xyz: tuple[int, int, int],
    roi_size_xyz: tuple[int, int, int],
) -> np.ndarray:
    sx, sy, sz = full_shape_xyz
    rx, ry, rz = roi_size_xyz
    x0 = (sx - rx) // 2
    y0 = (sy - ry) // 2
    z0 = (sz - rz) // 2

    crop_shift = np.eye(4, dtype=np.float64)
    crop_shift[0, 3] = x0
    crop_shift[1, 3] = y0
    crop_shift[2, 3] = z0

    roi_expected = np.linalg.inv(crop_shift) @ full_inverse_transform @ crop_shift
    return roi_expected[:3, :]


def plot_matrix_comparison(expected: np.ndarray, estimated: np.ndarray, out_path: Path) -> float:
    diff = estimated - expected
    frob = float(np.linalg.norm(diff))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    common_vmin = min(float(expected.min()), float(estimated.min()))
    common_vmax = max(float(expected.max()), float(estimated.max()))

    im0 = axes[0].imshow(expected, cmap="viridis", vmin=common_vmin, vmax=common_vmax)
    axes[0].set_title("Expected 3x4")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(estimated, cmap="viridis", vmin=common_vmin, vmax=common_vmax)
    axes[1].set_title("Estimated 3x4")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(np.abs(diff), cmap="magma")
    axes[2].set_title(f"|Difference|, Frobenius={frob:.4f}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return frob


def plot_noise_effect(clean_path: Path, noisy_path: Path, out_path: Path) -> dict[str, float]:
    clean = np.asarray(nib.load(str(clean_path)).get_fdata(dtype=np.float32), dtype=np.float32)
    noisy = np.asarray(nib.load(str(noisy_path)).get_fdata(dtype=np.float32), dtype=np.float32)
    diff = noisy - clean

    z = diff.shape[2] // 2
    diff_slice = diff[:, :, z]

    diff_mean = float(diff.mean())
    diff_std = float(diff.std())
    diff_min = float(diff.min())
    diff_max = float(diff.max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im = axes[0].imshow(diff_slice, cmap="coolwarm")
    axes[0].set_title(f"Central Slice Noise Map (z={z})")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].hist(diff.ravel(), bins=80, color="#3b5b92", alpha=0.9)
    axes[1].set_title(
        f"Noise Histogram\nmean={diff_mean:.3f}, std={diff_std:.3f}, "
        f"min={diff_min:.1f}, max={diff_max:.1f}"
    )
    axes[1].set_xlabel("Noisy - Clean")
    axes[1].set_ylabel("Voxel Count")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return {
        "noise_mean": diff_mean,
        "noise_std": diff_std,
        "noise_min": diff_min,
        "noise_max": diff_max,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact report plots for a single ROI/noise test case."
    )
    parser.add_argument("run_dir", type=Path, help="Run directory with base/noisy/results files.")
    parser.add_argument("--roi-size", nargs=3, type=int, metavar=("X", "Y", "Z"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    roi_size_xyz = tuple(args.roi_size)

    base_dir = run_dir
    source_stack_dir = run_dir.parent.parent / "resources" / "bhi_2_2.32um_voi"
    full_shape_xyz = stack_shape_xyz(source_stack_dir)

    expected_full_inv = read_transform_csv(base_dir / "T_full_inv_4x4.csv")
    expected_roi = roi_expected_transform(expected_full_inv, full_shape_xyz, roi_size_xyz)
    estimated = read_transform_csv(base_dir / "transform.csv")

    plots_dir = run_dir / "plots"
    frob = plot_matrix_comparison(expected_roi, estimated, plots_dir / "matrix_comparison.png")
    noise_stats = plot_noise_effect(
        run_dir / "volume_B_roi250_clean.nii",
        run_dir / "volume_B_roi250_noise_var_1000.nii",
        plots_dir / "noise_effect.png",
    )

    summary = {
        "run_dir": str(run_dir),
        "roi_size_xyz": list(roi_size_xyz),
        "matrix_frobenius_error": frob,
        **noise_stats,
    }
    (plots_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Plots dir: {plots_dir}")
    print(f"Matrix comparison: {plots_dir / 'matrix_comparison.png'}")
    print(f"Noise effect: {plots_dir / 'noise_effect.png'}")
    print(f"Summary: {plots_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
