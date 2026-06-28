from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from uir.v1.analysis.transform_metrics import (
    count_match_rows,
    infer_noisy_path,
    infer_requested_variance,
    match_residual_stats,
    matrix_error_stats,
    write_match_residuals_csv,
    write_matrix_element_errors_csv,
)
from uir.core.io.png_stack import inspect_png_stack
from uir.v1.reporting.single_case_plots import (
    plot_matrix_error_diagnostic,
    plot_match_residual_diagnostic,
    plot_noise_effect,
)
from uir.core.transforms.io import read_transform_csv, write_transform_csv
from uir.core.transforms.roi import roi_expected_transform


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
    parser.add_argument(
        "--noise-reference-path",
        type=Path,
        help="Path to the clean or blurred volume used as the AWGN input.",
    )
    parser.add_argument("--degradation", choices=("awgn", "blur_awgn"), help="Synthetic degradation chain.")
    parser.add_argument("--transform-tag", help="Synthetic transform tag.")
    parser.add_argument("--blur-sigma-xy", type=float, default=0.0, help="Gaussian blur sigma in XY slice units.")
    parser.add_argument("--awgn-variance", type=float, help="AWGN variance used for the run.")
    parser.add_argument("--awgn-seed", type=int, default=42, help="AWGN random seed used for the run.")
    parser.add_argument("--reg-exit-code", type=int, default=0, help="Exit code returned by regSift3D.")
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
    run_dir = args.run_dir
    roi_size_xyz = tuple(args.roi_size)

    source_stack_dir = args.source_stack_dir or (run_dir.parent.parent / "resources" / "bhi_2_2.32um_voi")
    _, full_shape_xyz, source_observation_model = inspect_png_stack(source_stack_dir)

    expected_full_inv = read_transform_csv(run_dir / "T_full_inv_4x4.csv")
    expected_roi = roi_expected_transform(expected_full_inv, full_shape_xyz, roi_size_xyz)
    noisy_path = args.noisy_path or infer_noisy_path(run_dir, roi_size=roi_size_xyz[0])
    matches_path = args.matches_path or (run_dir / "matches.csv")
    transform_path = run_dir / "transform.csv"
    noise_reference_path = args.noise_reference_path or (run_dir / f"volume_B_roi{roi_size_xyz[0]}_clean.nii")
    requested_variance = args.awgn_variance
    if requested_variance is None:
        requested_variance = infer_requested_variance(noisy_path)
    degradation = args.degradation or ("blur_awgn" if args.blur_sigma_xy > 0.0 else "awgn")
    transform_tag = args.transform_tag or run_dir.parent.name

    plots_dir = run_dir / "plots"
    expected_roi_path = run_dir / "expected_roi_transform.csv"
    diff_path = plots_dir / "transform_minus_expected.csv"
    element_errors_path = plots_dir / "matrix_element_errors.csv"
    diagnostic_path = plots_dir / "matrix_error_diagnostic.png"
    match_residuals_path = plots_dir / "match_residuals.csv"
    match_residual_diagnostic_path = plots_dir / "match_residual_diagnostic.png"
    summary_path = plots_dir / "summary.json"
    previous_summary = {}
    if summary_path.exists():
        previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    plots_dir.mkdir(parents=True, exist_ok=True)
    write_transform_csv(expected_roi_path, expected_roi)
    registration_succeeded = args.reg_exit_code == 0 and transform_path.exists()
    if registration_succeeded:
        estimated = read_transform_csv(transform_path)
        write_transform_csv(diff_path, estimated - expected_roi)
        write_matrix_element_errors_csv(element_errors_path, expected_roi, estimated)
        write_match_residuals_csv(match_residuals_path, matches_path, estimated)

        plot_matrix_error_diagnostic(expected_roi, estimated, diagnostic_path)
        plot_match_residual_diagnostic(matches_path, estimated, match_residual_diagnostic_path)
        error_stats = matrix_error_stats(expected_roi, estimated)
        residual_stats = match_residual_stats(matches_path, estimated)
    else:
        error_stats = _empty_error_stats()
        residual_stats = _empty_residual_stats()
    noise_keys = [
        "noise_mean",
        "noise_std",
        "noise_min",
        "noise_max",
        "clean_observation_min",
        "clean_observation_max",
        "noisy_observation_min",
        "noisy_observation_max",
    ]
    if noise_reference_path.exists() and noisy_path.exists():
        noise_stats = plot_noise_effect(
            noise_reference_path,
            noisy_path,
            plots_dir / "noise_effect.png",
            observation_min=source_observation_model.min_value,
            observation_max=source_observation_model.max_value,
        )
    else:
        missing_paths = [path for path in (noise_reference_path, noisy_path) if not path.exists()]
        previous_summary.setdefault("clean_observation_min", source_observation_model.min_value)
        previous_summary.setdefault("clean_observation_max", source_observation_model.max_value)
        previous_summary.setdefault("noisy_observation_min", source_observation_model.min_value)
        previous_summary.setdefault("noisy_observation_max", source_observation_model.max_value)
        missing_noise_keys = [key for key in noise_keys if key not in previous_summary]
        if missing_noise_keys:
            raise RuntimeError(
                f"Cannot rebuild noise stats because files are missing: {missing_paths}; "
                f"previous summary is missing keys: {missing_noise_keys}"
            )
        noise_stats = {key: previous_summary[key] for key in noise_keys}
        print(f"Keeping previous noise stats; missing files: {', '.join(str(path) for path in missing_paths)}")
    match_count = count_match_rows(matches_path)

    summary = {
        "run_kind": "synthetic",
        "registration_succeeded": registration_succeeded,
        "reg_exit_code": int(args.reg_exit_code),
        "degradation": degradation,
        "transform_tag": transform_tag,
        "run_dir": str(run_dir),
        "roi_size_xyz": list(roi_size_xyz),
        "blur_sigma_xy": float(args.blur_sigma_xy),
        "awgn_seed": int(args.awgn_seed),
        "expected_transform_path": str(expected_roi_path),
        "estimated_transform_path": str(transform_path),
        "transform_diff_path": str(diff_path),
        "matrix_element_errors_path": str(element_errors_path),
        "matrix_error_diagnostic_path": str(diagnostic_path),
        "match_residuals_path": str(match_residuals_path),
        "match_residual_diagnostic_path": str(match_residual_diagnostic_path),
        "noise_reference_path": str(noise_reference_path),
        "noisy_path": str(noisy_path),
        "matches_path": str(matches_path),
        "match_count": match_count,
        **error_stats,
        **residual_stats,
        **noise_stats,
    }
    if requested_variance is not None:
        expected_noise_std = math.sqrt(requested_variance)
        summary["awgn_variance"] = requested_variance
        summary["requested_variance"] = requested_variance
        summary["expected_noise_std"] = expected_noise_std
        summary["noise_std_abs_error"] = abs(noise_stats["noise_std"] - expected_noise_std)
        summary["noise_variance_observed"] = noise_stats["noise_std"] ** 2
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Plots dir: {plots_dir}")
    print(f"Registration succeeded: {registration_succeeded}")
    print(f"Expected ROI transform: {expected_roi_path}")
    print(f"Noisy ROI NIfTI: {noisy_path}")
    print(f"Matches CSV: {matches_path}")
    print(f"Match count: {match_count}")
    if registration_succeeded:
        print(f"Transform diff CSV: {diff_path}")
        print(f"Matrix element errors: {element_errors_path}")
        print(f"Matrix error diagnostic: {diagnostic_path}")
        print(f"Match residuals CSV: {match_residuals_path}")
        print(f"Match residual diagnostic: {match_residual_diagnostic_path}")
    print(f"Noise effect: {plots_dir / 'noise_effect.png'}")
    if registration_succeeded:
        print(f"Translation error XYZ: {error_stats['translation_error_xyz']}")
        print(f"Linear RMS error: {error_stats['linear_rms_error']:.6f}")
        print(f"Translation L2 error: {error_stats['translation_l2_error_voxels']:.6f} voxels")
        print(f"Match residual mean: {residual_stats['match_residual_l2_mean']:.6f} voxels")
        print(f"Match residual p95: {residual_stats['match_residual_l2_p95']:.6f} voxels")
    if requested_variance is not None:
        print(f"Requested variance: {requested_variance:.0f}")
        print(f"Observed noise std: {noise_stats['noise_std']:.6f}")
        print(f"Expected noise std: {math.sqrt(requested_variance):.6f}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
