from __future__ import annotations

import argparse
import json
import math
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


def write_transform_csv(path: Path, mat: np.ndarray) -> None:
    path.write_text(
        "\n".join(
            ",".join(f"{float(value):.12f}" for value in row)
            for row in mat
        )
        + "\n",
        encoding="utf-8",
    )


def infer_requested_variance(noisy_path: Path) -> float | None:
    m = re.search(r"_var_(\d+)", noisy_path.name)
    if m is None:
        return None
    return float(int(m.group(1)))


def infer_noisy_path(run_dir: Path, roi_size: int = 250) -> Path:
    candidates = sorted(run_dir.glob(f"volume_B_roi{roi_size}_noise_var_*.nii*"))
    if not candidates:
        raise RuntimeError(f"Cannot find noisy ROI NIfTI in {run_dir}")
    return candidates[0]


def count_match_rows(matches_path: Path) -> int:
    if not matches_path.exists():
        return 0
    with matches_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


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


def matrix_error_stats(expected: np.ndarray, estimated: np.ndarray) -> dict[str, object]:
    diff = estimated - expected
    linear_diff = diff[:, :3]
    translation_diff = diff[:, 3]
    expected_det = float(np.linalg.det(expected[:, :3]))
    estimated_det = float(np.linalg.det(estimated[:, :3]))

    return {
        "matrix_frobenius_error": float(np.linalg.norm(diff)),
        "linear_frobenius_error": float(np.linalg.norm(linear_diff)),
        "translation_l2_error": float(np.linalg.norm(translation_diff)),
        "max_abs_matrix_error": float(np.max(np.abs(diff))),
        "max_abs_linear_error": float(np.max(np.abs(linear_diff))),
        "max_abs_translation_error": float(np.max(np.abs(translation_diff))),
        "translation_error_xyz": [float(v) for v in translation_diff],
        "expected_linear_det": expected_det,
        "estimated_linear_det": estimated_det,
        "linear_det_abs_error": float(abs(estimated_det - expected_det)),
    }


def plot_matrix_error_detail(expected: np.ndarray, estimated: np.ndarray, out_path: Path) -> None:
    diff = estimated - expected
    linear_diff = diff[:, :3]
    translation_diff = diff[:, 3]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    im = axes[0].imshow(linear_diff, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[0].set_title("Linear Diff Clipped To [-1, 1]")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].bar(["Tx", "Ty", "Tz"], translation_diff, color=["#3b5b92", "#5f8f3b", "#b55d42"])
    axes[1].axhline(1.0, color="#666666", linestyle="--", linewidth=1)
    axes[1].axhline(-1.0, color="#666666", linestyle="--", linewidth=1)
    axes[1].set_title("Translation Error")
    axes[1].set_ylabel("Estimated - Expected")
    axes[1].grid(True, axis="y", alpha=0.3)

    for i, value in enumerate(translation_diff):
        axes[1].text(i, value, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


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
    parser.add_argument(
        "--source-stack-dir",
        type=Path,
        help="Path to the original PNG stack used as the uncropped source volume.",
    )
    parser.add_argument(
        "--noisy-path",
        type=Path,
        help="Path to the noisy ROI NIfTI used for the comparison plots.",
    )
    parser.add_argument(
        "--matches-path",
        type=Path,
        help="Path to the CSV file with matched features.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    roi_size_xyz = tuple(args.roi_size)

    base_dir = run_dir
    source_stack_dir = args.source_stack_dir or (run_dir.parent.parent / "resources" / "bhi_2_2.32um_voi")
    full_shape_xyz = stack_shape_xyz(source_stack_dir)

    expected_full_inv = read_transform_csv(base_dir / "T_full_inv_4x4.csv")
    expected_roi = roi_expected_transform(expected_full_inv, full_shape_xyz, roi_size_xyz)
    estimated = read_transform_csv(base_dir / "transform.csv")
    noisy_path = args.noisy_path or infer_noisy_path(run_dir, roi_size=roi_size_xyz[0])
    matches_path = args.matches_path or (run_dir / "matches.csv")
    requested_variance = infer_requested_variance(noisy_path)

    plots_dir = run_dir / "plots"
    expected_roi_path = run_dir / "expected_roi_transform.csv"
    expected_reg_path = run_dir / "expected_reg_transform.csv"
    diff_path = plots_dir / "transform_minus_expected.csv"

    plots_dir.mkdir(parents=True, exist_ok=True)
    write_transform_csv(expected_roi_path, expected_roi)
    write_transform_csv(expected_reg_path, expected_roi)
    write_transform_csv(diff_path, estimated - expected_roi)

    frob = plot_matrix_comparison(expected_roi, estimated, plots_dir / "matrix_comparison.png")
    plot_matrix_error_detail(expected_roi, estimated, plots_dir / "matrix_error_detail.png")
    error_stats = matrix_error_stats(expected_roi, estimated)
    noise_stats = plot_noise_effect(
        run_dir / f"volume_B_roi{roi_size_xyz[0]}_clean.nii",
        noisy_path,
        plots_dir / "noise_effect.png",
    )
    match_count = count_match_rows(matches_path)

    summary = {
        "run_dir": str(run_dir),
        "roi_size_xyz": list(roi_size_xyz),
        "expected_transform_path": str(expected_roi_path),
        "estimated_transform_path": str(run_dir / "transform.csv"),
        "transform_diff_path": str(diff_path),
        "noisy_path": str(noisy_path),
        "matches_path": str(matches_path),
        "match_count": match_count,
        "matrix_frobenius_error": frob,
        **error_stats,
        **noise_stats,
    }
    if requested_variance is not None:
        expected_noise_std = math.sqrt(requested_variance)
        summary["requested_variance"] = requested_variance
        summary["expected_noise_std"] = expected_noise_std
        summary["noise_std_abs_error"] = abs(noise_stats["noise_std"] - expected_noise_std)
        summary["noise_variance_observed"] = noise_stats["noise_std"] ** 2
    (plots_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Plots dir: {plots_dir}")
    print(f"Expected ROI transform: {expected_roi_path}")
    print(f"Noisy ROI NIfTI: {noisy_path}")
    print(f"Matches CSV: {matches_path}")
    print(f"Match count: {match_count}")
    print(f"Transform diff CSV: {diff_path}")
    print(f"Matrix comparison: {plots_dir / 'matrix_comparison.png'}")
    print(f"Matrix error detail: {plots_dir / 'matrix_error_detail.png'}")
    print(f"Noise effect: {plots_dir / 'noise_effect.png'}")
    print(f"Translation error XYZ: {error_stats['translation_error_xyz']}")
    print(f"Linear Frobenius error: {error_stats['linear_frobenius_error']:.6f}")
    print(f"Translation L2 error: {error_stats['translation_l2_error']:.6f}")
    if requested_variance is not None:
        print(f"Requested variance: {requested_variance:.0f}")
        print(f"Observed noise std: {noise_stats['noise_std']:.6f}")
        print(f"Expected noise std: {math.sqrt(requested_variance):.6f}")
    print(f"Summary: {plots_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
