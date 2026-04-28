from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_synthetic_metric(
    rows: list[dict[str, object]],
    out_path: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    groups: dict[tuple[object, object, object, object], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            (
                row.get("transform_id"),
                row.get("roi_size"),
                row.get("degradation"),
                row.get("blur_sigma_xy"),
            ),
            [],
        ).append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    for (transform_id, roi_size, degradation, blur_sigma_xy), group_rows in sorted(groups.items()):
        points = [
            (float(row["awgn_variance"]), float(row[metric]))
            for row in group_rows
            if row.get(metric) not in (None, "")
        ]
        if not points:
            continue
        points.sort()
        xs, ys = zip(*points)
        label = f"{transform_id} roi{roi_size} {degradation} blur={float(blur_sigma_xy):g}"
        ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=4, label=label)

    ax.set_title(title)
    ax.set_xlabel("AWGN variance")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


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
