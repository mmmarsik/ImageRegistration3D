from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from uir.analysis.transform_metrics import (
    count_match_rows,
    match_residual_stats,
    matrix_error_stats,
    write_match_residuals_csv,
    write_matrix_element_errors_csv,
)
from uir.reporting.single_case_plots import plot_match_residual_diagnostic, plot_matrix_error_diagnostic
from uir.transforms.io import read_transform_csv, write_transform_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one physical-resolution crop registration case.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--mode", choices=("resample",), required=True)
    parser.add_argument("--matches-path", type=Path, required=True)
    parser.add_argument("--transform-path", type=Path, required=True)
    parser.add_argument("--expected-transform-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--reg-exit-code", type=int, required=True)
    return parser.parse_args()


def _empty_error_stats() -> dict[str, object]:
    return {
        "linear_rms_error": None,
        "linear_mean_abs_error": None,
        "linear_max_abs_error": None,
        "linear_max_abs_error_component": None,
        "translation_l2_error_voxels": None,
        "translation_mean_abs_error_voxels": None,
        "translation_max_abs_error_voxels": None,
        "translation_max_abs_error_axis": None,
        "max_abs_transform_element_error": None,
        "max_abs_transform_element_component": None,
        "translation_error_xyz": None,
        "expected_linear_det": None,
        "estimated_linear_det": None,
        "linear_det_abs_error": None,
    }


def _empty_residual_stats() -> dict[str, object]:
    return {
        "match_residual_count": 0,
        "match_raw_l2_mean": None,
        "match_raw_l2_median": None,
        "match_raw_l2_max": None,
        "match_residual_l2_mean": None,
        "match_residual_l2_median": None,
        "match_residual_l2_rms": None,
        "match_residual_l2_p95": None,
        "match_residual_l2_max": None,
        "match_residual_xyz_mean": None,
    }


def main() -> int:
    args = parse_args()
    metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    mode_dir = args.run_dir / args.mode
    plots_dir = mode_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    expected = read_transform_csv(args.expected_transform_path)
    match_count = count_match_rows(args.matches_path)
    registration_succeeded = args.reg_exit_code == 0 and args.transform_path.exists()

    transform_diff_path = plots_dir / "transform_minus_expected.csv"
    element_errors_path = plots_dir / "matrix_element_errors.csv"
    matrix_plot_path = plots_dir / "matrix_error_diagnostic.png"
    residuals_path = plots_dir / "match_residuals.csv"
    residual_plot_path = plots_dir / "match_residual_diagnostic.png"

    if registration_succeeded:
        estimated = read_transform_csv(args.transform_path)
        diff = estimated - expected
        write_transform_csv(transform_diff_path, diff)
        write_matrix_element_errors_csv(element_errors_path, expected, estimated)
        plot_matrix_error_diagnostic(expected, estimated, matrix_plot_path)
        write_match_residuals_csv(residuals_path, args.matches_path, estimated)
        plot_match_residual_diagnostic(args.matches_path, estimated, residual_plot_path)
        error_stats = matrix_error_stats(expected, estimated)
        residual_stats = match_residual_stats(args.matches_path, estimated)
        low_spacing = float(metadata["low_spacing"])
        translation_error_xyz = error_stats.get("translation_error_xyz")
        if isinstance(translation_error_xyz, list):
            error_stats["translation_error_physical_xyz"] = [float(v) * low_spacing for v in translation_error_xyz]
        if error_stats.get("translation_l2_error_voxels") is not None:
            error_stats["translation_l2_error_physical"] = float(error_stats["translation_l2_error_voxels"]) * low_spacing
    else:
        error_stats = _empty_error_stats()
        residual_stats = _empty_residual_stats()
        error_stats["translation_error_physical_xyz"] = None
        error_stats["translation_l2_error_physical"] = None

    summary = {
        **metadata,
        "mode": args.mode,
        "registration_succeeded": registration_succeeded,
        "reg_exit_code": int(args.reg_exit_code),
        "run_dir": str(args.run_dir),
        "mode_dir": str(mode_dir),
        "matches_path": str(args.matches_path),
        "transform_path": str(args.transform_path),
        "match_count": match_count,
        "transform_diff_path": str(transform_diff_path),
        "matrix_element_errors_path": str(element_errors_path),
        "matrix_error_diagnostic_path": str(matrix_plot_path),
        "match_residuals_path": str(residuals_path),
        "match_residual_diagnostic_path": str(residual_plot_path),
        **error_stats,
        **residual_stats,
    }
    summary_path = mode_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Mode: {args.mode}")
    print(f"Registration succeeded: {registration_succeeded}")
    print(f"Match count: {match_count}")
    if registration_succeeded:
        print(f"Translation L2 error: {summary['translation_l2_error_voxels']:.6f} low-res voxels")
        print(f"Translation L2 physical error: {summary['translation_l2_error_physical']:.6f}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
