from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from uir.v1.analysis.transform_metrics import (
    apply_transform_to_points,
    read_match_points,
)


def plot_matrix_error_diagnostic(expected: np.ndarray, estimated: np.ndarray, out_path: Path) -> None:
    diff = estimated - expected
    limit = max(float(np.max(np.abs(diff))), 1e-12)
    linear_rms = float(np.sqrt(np.mean(diff[:, :3] * diff[:, :3])))
    translation_l2 = float(np.linalg.norm(diff[:, 3]))
    max_abs = float(np.max(np.abs(diff)))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    im = ax.imshow(diff, cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_title(
        "Ошибка элементов матрицы: оценка - эталон\n"
        f"линейная RMS={linear_rms:.6g}, L2 смещения={translation_l2:.6g}, max abs={max_abs:.6g}"
    )
    ax.set_xticks(range(4), labels=["c0", "c1", "c2", "смещение"])
    ax.set_yticks(range(3), labels=["строка 0", "строка 1", "строка 2"])
    ax.set_xlabel("Столбец преобразования")
    ax.set_ylabel("Строка преобразования")

    for row in range(diff.shape[0]):
        for col in range(diff.shape[1]):
            value = diff[row, col]
            text_color = "white" if abs(float(value)) > 0.55 * limit else "black"
            ax.text(col, row, f"{value:.4g}", ha="center", va="center", color=text_color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Оценка - эталон")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_match_residual_diagnostic(matches_path: Path, transform: np.ndarray, out_path: Path) -> dict[str, float]:
    source_xyz, reference_xyz = read_match_points(matches_path)
    if source_xyz.shape[0] == 0:
        raise RuntimeError(f"No matches found in {matches_path}")

    predicted_source_xyz = apply_transform_to_points(transform, reference_xyz)
    residual_xyz = source_xyz - predicted_source_xyz
    residual_l2 = np.linalg.norm(residual_xyz, axis=1)
    raw_l2 = np.linalg.norm(source_xyz - reference_xyz, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bins = min(80, max(12, int(np.sqrt(residual_l2.size))))
    axes[0].hist(raw_l2, bins=bins, alpha=0.55, label="До преобразования", color="#777777")
    axes[0].hist(residual_l2, bins=bins, alpha=0.75, label="После преобразования", color="#2f6db3")
    axes[0].set_title(
        "Расстояния между сопоставленными точками\n"
        f"после: среднее={np.mean(residual_l2):.3f}, медиана={np.median(residual_l2):.3f}, "
        f"p95={np.percentile(residual_l2, 95):.3f}"
    )
    axes[0].set_xlabel("Расстояние, воксели")
    axes[0].set_ylabel("Количество совпадений")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    order = np.argsort(residual_l2)
    axes[1].plot(residual_l2[order], color="#2f6db3", linewidth=1.2)
    axes[1].set_title(f"Отсортированные невязки (n={residual_l2.size})")
    axes[1].set_xlabel("Индекс совпадения после сортировки")
    axes[1].set_ylabel("|источник - T(эталон)|")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return {
        "match_raw_l2_mean": float(np.mean(raw_l2)),
        "match_raw_l2_median": float(np.median(raw_l2)),
        "match_raw_l2_max": float(np.max(raw_l2)),
        "match_residual_l2_mean": float(np.mean(residual_l2)),
        "match_residual_l2_median": float(np.median(residual_l2)),
        "match_residual_l2_rms": float(np.sqrt(np.mean(residual_l2 * residual_l2))),
        "match_residual_l2_p95": float(np.percentile(residual_l2, 95)),
        "match_residual_l2_max": float(np.max(residual_l2)),
    }


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
    axes[0].set_title(f"Карта шума центрального среза (z={z})")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].hist(diff.ravel(), bins=80, color="#3b5b92", alpha=0.9)
    axes[1].set_title(
        f"Гистограмма шума\nсреднее={diff_mean:.3f}, СКО={diff_std:.3f}, "
        f"min={diff_min:.1f}, max={diff_max:.1f}"
    )
    axes[1].set_xlabel("Зашумленный - чистый")
    axes[1].set_ylabel("Количество вокселей")
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
