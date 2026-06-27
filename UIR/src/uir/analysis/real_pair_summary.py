from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from uir.analysis.transform_metrics import (
    count_match_rows,
    match_residual_stats,
    write_match_residuals_csv,
)
from uir.io.png_stack import inspect_png_stack
from uir.reporting.single_case_plots import plot_match_residual_diagnostic
from uir.reporting.real_pair_stacks import (
    INTENSITY_CUBOID_RADIUS_DEFAULT,
    MODEL_CONSISTENT_THRESHOLD_DEFAULT,
    save_matched_keypoint_stacks,
    save_signed_diff_stack,
)
from uir.transforms.io import read_transform_csv


def summarize_real_pair_case(
    *,
    run_dir: Path,
    before_stack_dir: Path,
    after_stack_dir: Path,
    before_nifti: Path,
    after_nifti: Path,
    matches_path: Path,
    transform_path: Path,
    reg_exit_code: int = 0,
    roi_size_xyz: tuple[int, int, int] | None = None,
    before_roi_start_xyz: tuple[int, int, int] | None = None,
    after_roi_start_xyz: tuple[int, int, int] | None = None,
    model_consistent_threshold: float = MODEL_CONSISTENT_THRESHOLD_DEFAULT,
    intensity_cuboid_radius: int = INTENSITY_CUBOID_RADIUS_DEFAULT,
) -> dict[str, object]:
    _, before_shape_xyz, before_observation_model = inspect_png_stack(before_stack_dir)
    _, after_shape_xyz, after_observation_model = inspect_png_stack(after_stack_dir)

    before_img = nib.load(str(before_nifti))
    after_img = nib.load(str(after_nifti))

    summary: dict[str, object] = {
        "mode": "real_pair",
        "registration_succeeded": reg_exit_code == 0,
        "reg_exit_code": int(reg_exit_code),
        "run_dir": str(run_dir),
        "before_stack_dir": str(before_stack_dir),
        "after_stack_dir": str(after_stack_dir),
        "before_nifti": str(before_nifti),
        "after_nifti": str(after_nifti),
        "matches_path": str(matches_path),
        "transform_path": str(transform_path),
        "match_count": count_match_rows(matches_path),
        "before_stack_shape_xyz": list(before_shape_xyz),
        "after_stack_shape_xyz": list(after_shape_xyz),
        "before_nifti_shape_xyz": [int(v) for v in before_img.shape],
        "after_nifti_shape_xyz": [int(v) for v in after_img.shape],
        "before_observation_range": [
            float(before_observation_model.min_value),
            float(before_observation_model.max_value),
        ],
        "after_observation_range": [
            float(after_observation_model.min_value),
            float(after_observation_model.max_value),
        ],
    }
    plots_dir = run_dir / "plots"

    if transform_path.exists():
        estimated = read_transform_csv(transform_path)
        linear = estimated[:, :3]
        match_residuals_path = plots_dir / "match_residuals.csv"
        match_residual_diagnostic_path = plots_dir / "match_residual_diagnostic.png"
        if matches_path.exists() and summary["match_count"] > 0:
            summary.update(
                save_matched_keypoint_stacks(
                    before_nifti=before_nifti,
                    after_nifti=after_nifti,
                    matches_path=matches_path,
                    transform=estimated,
                    model_consistent_threshold=model_consistent_threshold,
                    before_out_dir=plots_dir / "matched_keypoints_before_png_stack",
                    after_out_dir=plots_dir / "matched_keypoints_after_png_stack",
                )
            )
            write_match_residuals_csv(match_residuals_path, matches_path, estimated)
            plot_match_residual_diagnostic(matches_path, estimated, match_residual_diagnostic_path)
            summary.update(match_residual_stats(matches_path, estimated))
            summary["match_residuals_path"] = str(match_residuals_path)
            summary["match_residual_diagnostic_path"] = str(match_residual_diagnostic_path)
        summary.update(
            save_signed_diff_stack(
                before_nifti=before_nifti,
                after_nifti=after_nifti,
                transform=estimated,
                matches_path=matches_path,
                model_consistent_threshold=model_consistent_threshold,
                intensity_cuboid_radius=intensity_cuboid_radius,
                out_dir=plots_dir / "signed_diff_png_stack",
            )
        )
        summary["estimated_transform_rows"] = int(estimated.shape[0])
        summary["estimated_transform_cols"] = int(estimated.shape[1])
        summary["estimated_linear_det"] = float(np.linalg.det(linear))

    if roi_size_xyz is not None:
        summary["roi_size_xyz"] = [int(v) for v in roi_size_xyz]
    if before_roi_start_xyz is not None:
        summary["before_roi_start_xyz"] = [int(v) for v in before_roi_start_xyz]
    if after_roi_start_xyz is not None:
        summary["after_roi_start_xyz"] = [int(v) for v in after_roi_start_xyz]

    return summary
