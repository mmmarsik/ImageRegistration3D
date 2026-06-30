from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uir.core.transforms.io import read_transform_csv
from uir.core.transforms.metrics import (
    apply_transform_to_points,
    count_match_rows,
    match_residual_stats,
    read_match_points,
    write_match_residuals_csv,
)

MODEL_CONSISTENT_THRESHOLD_DEFAULT = 5.0


def model_consistent_stats(
    matches_path: Path,
    transform: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    source_xyz, reference_xyz = read_match_points(matches_path)
    predicted = apply_transform_to_points(transform, reference_xyz)
    residual_l2 = np.linalg.norm(source_xyz - predicted, axis=1)

    if residual_l2.size == 0:
        return {
            "model_consistent_threshold": float(threshold),
            "model_consistent_count": 0,
            "model_consistent_fraction": None,
            "model_consistent_residual_l2_rms": None,
            "model_consistent_residual_l2_median": None,
            "model_consistent_residual_l2_p95": None,
            "model_consistent_residual_l2_max": None,
        }

    mask = residual_l2 <= threshold
    inliers = residual_l2[mask]
    stats: dict[str, object] = {
        "model_consistent_threshold": float(threshold),
        "model_consistent_count": int(mask.sum()),
        "model_consistent_fraction": float(mask.mean()),
        "model_consistent_residual_l2_rms": None,
        "model_consistent_residual_l2_median": None,
        "model_consistent_residual_l2_p95": None,
        "model_consistent_residual_l2_max": None,
    }
    if inliers.size > 0:
        stats.update(
            {
                "model_consistent_residual_l2_rms": float(np.sqrt(np.mean(inliers * inliers))),
                "model_consistent_residual_l2_median": float(np.median(inliers)),
                "model_consistent_residual_l2_p95": float(np.percentile(inliers, 95)),
                "model_consistent_residual_l2_max": float(np.max(inliers)),
            }
        )
    return stats


def evaluate_run(
    out_dir: Path,
    threshold: float = MODEL_CONSISTENT_THRESHOLD_DEFAULT,
) -> dict[str, object]:
    out_dir = Path(out_dir)
    matches_path = out_dir / "matches.csv"
    transform_path = out_dir / "transform.csv"

    if not transform_path.exists():
        raise RuntimeError(f"transform.csv not found in {out_dir}")

    transform = read_transform_csv(transform_path)
    residuals_path = out_dir / "match_residuals.csv"
    write_match_residuals_csv(residuals_path, matches_path, transform)

    metrics: dict[str, object] = {
        "match_count": count_match_rows(matches_path),
        **match_residual_stats(matches_path, transform),
        **model_consistent_stats(matches_path, transform, threshold),
        "match_residuals_path": str(residuals_path),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
