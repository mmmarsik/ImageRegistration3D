from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np



def plot_real_pair_roi_summary(rows: list[dict[str, object]], out_path: Path) -> None:
    plot_rows = sorted(rows, key=lambda row: int(row["roi_size"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    error_points = [
        (int(row["roi_size"]), float(row["translation_l2_error_voxels"]))
        for row in plot_rows
        if row.get("translation_l2_error_voxels") not in (None, "")
    ]
    if error_points:
        xs, ys = zip(*error_points)
        axes[0].plot(xs, ys, marker="o", linewidth=1.8)
    axes[0].set_title("ROI Translation Error vs Full Volume")
    axes[0].set_xlabel("ROI edge size")
    axes[0].set_ylabel("Relative-to-full translation L2 error (voxels)")
    axes[0].grid(True, alpha=0.3)

    match_points = [(int(row["roi_size"]), int(row.get("match_count") or 0)) for row in plot_rows]
    if match_points:
        xs, ys = zip(*match_points)
        axes[1].plot(xs, ys, marker="o", linewidth=1.8, color="#3b5b92")
    axes[1].set_title("Matched Points vs ROI Size")
    axes[1].set_xlabel("ROI edge size")
    axes[1].set_ylabel("match_count")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_noise_sweep(
    rows: list[dict[str, object]],
    out_path: Path,
    *,
    roi_size: int,
    metric: str = "translation_l2_error_voxels",
    ylabel: str = "Translation L2 error (voxels)",
) -> None:
    subset = [r for r in rows if int(r["roi_size"]) == roi_size]
    if not subset:
        return

    groups: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in subset:
        groups[(str(row["transform_id"]), float(row["blur_sigma_xy"]))].append(row)

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"ROI {roi_size}: Effect of Noise Level", fontsize=12)

    for i, (transform_id, blur_sigma) in enumerate(sorted(groups)):
        group_rows = sorted(groups[(transform_id, blur_sigma)], key=lambda r: float(r["awgn_variance"]))
        xs = [float(r["awgn_variance"]) for r in group_rows]
        label = f"{transform_id} blur={blur_sigma:g}"
        color = colors[i % len(colors)]

        ys_err = [float(r[metric]) for r in group_rows if r.get(metric) not in (None, "")]
        if ys_err:
            axes[0].plot(xs[: len(ys_err)], ys_err, marker="o", linewidth=1.8, markersize=5, label=label, color=color)

        ys_mc = [int(r["match_count"]) for r in group_rows if r.get("match_count") not in (None, "")]
        if ys_mc:
            axes[1].plot(xs[: len(ys_mc)], ys_mc, marker="o", linewidth=1.8, markersize=5, label=label, color=color)

    axes[0].set_title(f"{ylabel} vs Noise")
    axes[0].set_xlabel("AWGN variance")
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Match Count vs Noise")
    axes[1].set_xlabel("AWGN variance")
    axes[1].set_ylabel("Match count")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_error_heatmap(rows: list[dict[str, object]], out_path: Path) -> None:
    cell_data: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        v = row.get("translation_l2_error_voxels")
        if v not in (None, ""):
            cell_data[(str(row["transform_id"]), int(row["roi_size"]))].append(float(v))

    transform_ids = sorted({k[0] for k in cell_data})
    roi_sizes = sorted({k[1] for k in cell_data})
    if not transform_ids or not roi_sizes:
        return

    matrix = np.full((len(transform_ids), len(roi_sizes)), np.nan)
    for i, tid in enumerate(transform_ids):
        for j, rsz in enumerate(roi_sizes):
            vals = cell_data.get((tid, rsz), [])
            if vals:
                matrix[i, j] = float(np.median(vals))

    fig, ax = plt.subplots(figsize=(max(6, len(roi_sizes) * 1.8), max(4, len(transform_ids) * 1.0)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(roi_sizes)))
    ax.set_xticklabels([str(s) for s in roi_sizes])
    ax.set_yticks(range(len(transform_ids)))
    ax.set_yticklabels(transform_ids)
    ax.set_xlabel("ROI size (voxels)")
    ax.set_ylabel("Transform")
    ax.set_title("Median Translation L2 Error (voxels): Transform × ROI Size")

    vmax = float(np.nanmax(matrix)) if not np.all(np.isnan(matrix)) else 1.0
    for i in range(len(transform_ids)):
        for j in range(len(roi_sizes)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.6 * vmax else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label="Median translation L2 error (voxels)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def plot_reliability_curve(rows: list[dict[str, object]], out_path: Path) -> None:
    groups: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        v = row.get("translation_l2_error_voxels")
        if v not in (None, ""):
            groups[int(row["roi_size"])].append(float(v))

    if not groups:
        return

    all_errors = [v for vals in groups.values() for v in vals]
    thresholds = np.linspace(0.0, float(np.max(all_errors)), 400)

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, roi_size in enumerate(sorted(groups)):
        errors = np.array(sorted(groups[roi_size]))
        fracs = np.array([(errors <= t).mean() for t in thresholds])
        ax.plot(thresholds, fracs, linewidth=1.8, color=colors[i % len(colors)], label=f"roi={roi_size}")

    ax.set_title("Reliability Curve: Fraction of Runs Within Error Threshold")
    ax.set_xlabel("Translation L2 error threshold (voxels)")
    ax.set_ylabel("Fraction of runs ≤ threshold")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def plot_noise_accuracy(rows: list[dict[str, object]], out_path: Path) -> None:
    points: list[tuple[float, float]] = []
    for row in rows:
        expected = row.get("expected_noise_std")
        observed = row.get("noise_std")
        if expected not in (None, "") and observed not in (None, ""):
            try:
                points.append((float(expected), float(observed)))
            except (ValueError, TypeError):
                pass

    if not points:
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    lo = min(xs + ys) * 0.95
    hi = max(xs + ys) * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, s=20, alpha=0.6, color="#3b5b92")
    ax.plot([lo, hi], [lo, hi], "--", color="#aaaaaa", linewidth=1.2, label="ideal (y=x)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("Noise Accuracy: Expected vs Observed Noise Std")
    ax.set_xlabel("Expected noise std")
    ax.set_ylabel("Observed noise std")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
