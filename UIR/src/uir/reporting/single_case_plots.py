from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def plot_matrix_error_diagnostic(expected: np.ndarray, estimated: np.ndarray, out_path: Path) -> None:
    diff = estimated - expected
    limit = max(float(np.max(np.abs(diff))), 1e-12)
    linear_rms = float(np.sqrt(np.mean(diff[:, :3] * diff[:, :3])))
    translation_l2 = float(np.linalg.norm(diff[:, 3]))
    max_abs = float(np.max(np.abs(diff)))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    im = ax.imshow(diff, cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_title(
        "Transform Element Error: Estimated - Expected\n"
        f"linear RMS={linear_rms:.6g}, translation L2={translation_l2:.6g}, max abs={max_abs:.6g}"
    )
    ax.set_xticks(range(4), labels=["c0", "c1", "c2", "translation"])
    ax.set_yticks(range(3), labels=["row 0", "row 1", "row 2"])
    ax.set_xlabel("Transform column")
    ax.set_ylabel("Transform row")

    for row in range(diff.shape[0]):
        for col in range(diff.shape[1]):
            value = diff[row, col]
            text_color = "white" if abs(float(value)) > 0.55 * limit else "black"
            ax.text(col, row, f"{value:.4g}", ha="center", va="center", color=text_color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Estimated - Expected")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_noise_effect(
    clean_path: Path,
    noisy_path: Path,
    out_path: Path,
    *,
    observation_min: float,
    observation_max: float,
) -> dict[str, float]:
    clean_img = nib.load(str(clean_path))
    noisy_img = nib.load(str(noisy_path))
    clean = np.asarray(clean_img.get_fdata(dtype=np.float32), dtype=np.float32)
    noisy = np.asarray(noisy_img.get_fdata(dtype=np.float32), dtype=np.float32)
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
        "clean_observation_min": observation_min,
        "clean_observation_max": observation_max,
        "noisy_observation_min": observation_min,
        "noisy_observation_max": observation_max,
    }
