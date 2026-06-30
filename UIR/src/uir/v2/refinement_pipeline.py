from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

from uir.core.transforms.io import read_transform_csv, write_transform_csv
from uir.core.transforms.matrix import as_homogeneous_4x4
from uir.core.transforms.metrics import matrix_error_stats, write_matrix_element_errors_csv
from uir.registration.regsift3d import RegSift3D
from uir.v2.affine_pipeline import default_cases
from uir.v2.io import save_nifti_v2
from uir.v2.metrics import evaluate_run


DEFAULT_REFINE_CASE_IDS = [
    "t02_translation_medium",
    "t03_rotation_z_small",
    "t05_rotation_xyz_medium",
    "t06_scale_up_small",
    "t09_rotation_translation",
    "t10_combo_hard",
]


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_volume_xyz(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)


def warp_reference_grid_to_source(
    source_volume_xyz: np.ndarray,
    reference_to_source_transform: np.ndarray,
    *,
    output_shape: tuple[int, int, int],
) -> np.ndarray:
    """Return source sampled on the reference grid using a reference->source matrix."""

    mat = as_homogeneous_4x4(reference_to_source_transform)
    return affine_transform(
        source_volume_xyz,
        matrix=mat[:3, :3],
        offset=mat[:3, 3],
        output_shape=output_shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)


def _case_lookup() -> dict[str, object]:
    return {case.case_id: case for case in default_cases()}


def _baseline_rows(base_run_dir: Path) -> dict[tuple[str, str], dict[str, object]]:
    path = base_run_dir / "case_metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        return {(row["mode"], row["case_id"]): row for row in csv.DictReader(f)}


def _float_from_row(row: dict[str, object] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    if value is None or value == "":
        return None
    return float(value)


def _make_report(summary: dict[str, object], out_dir: Path) -> None:
    rows = list(summary["rows"])
    lines = [
        "# v2 coarse-to-local refinement report",
        "",
        "## Pipeline",
        "",
        f"- base run: `{summary['base_run_dir']}`",
        "- coarse matrix: `full_downsample/<case>/transform.csv` from the base run (`moving -> fixed`)",
        "- local data: `center_cube/<case>/moving.nii` and `center_cube/fixed.nii` from the base run",
        "- prewarp: sample original moving on the fixed grid using `inverse(coarse)`",
        "- residual registration: regSift3D on `fixed.nii` vs `moving_prewarped.nii`",
        "- final matrix: `T_final = T_residual @ T_coarse`",
        "",
        "## Results",
        "",
        "| case | full err | center err | refined err | refined-center | residual p95 | matches | time, s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {full_err} | {center_err} | {refined_err} | {delta} | {p95} | {matches} | {seconds:.2f} |".format(
                case_id=row["case_id"],
                full_err=_fmt(row.get("baseline_full_max_abs_transform_element_error")),
                center_err=_fmt(row.get("baseline_center_max_abs_transform_element_error")),
                refined_err=_fmt(row.get("refined_max_abs_transform_element_error")),
                delta=_fmt(row.get("refined_minus_center_max_abs_error")),
                p95=_fmt(row.get("residual_match_residual_l2_p95")),
                matches=row.get("residual_match_count", ""),
                seconds=float(row.get("registration_seconds", 0.0)),
            )
        )

    improved = [
        row for row in rows
        if row.get("refined_minus_center_max_abs_error") is not None
        and float(row["refined_minus_center_max_abs_error"]) < 0.0
    ]
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- refined better than direct center-cube on `{len(improved)} / {len(rows)}` cases by max matrix element error.",
            "- `residual p95` is computed on residual-registration matches, not on all voxels.",
            "- This is not a native regSift3D initial transform; the approximation is injected by prewarping the moving volume first.",
            "",
            "## Outputs",
            "",
            "- `summary.json`",
            "- `refinement_metrics.csv`",
            "- per-case `coarse_transform.csv`, `residual_transform.csv`, `refined_transform.csv`, `matrix_error.json`",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: object) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6g}"


def run_refinement_pipeline(
    *,
    base_run_dir: Path,
    out_dir: Path,
    binary: Path,
    case_ids: list[str] | None = None,
    omp_num_threads: int | None = None,
    skip_registration: bool = False,
) -> dict[str, object]:
    base_run_dir = Path(base_run_dir)
    out_dir = Path(out_dir)
    binary = Path(binary)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_case_ids = list(DEFAULT_REFINE_CASE_IDS if case_ids is None else case_ids)
    cases = _case_lookup()
    unknown = [case_id for case_id in selected_case_ids if case_id not in cases]
    if unknown:
        raise ValueError(f"unknown case ids: {unknown}")

    fixed_path = base_run_dir / "center_cube" / "fixed.nii"
    if not fixed_path.exists():
        raise RuntimeError(f"fixed center cube not found: {fixed_path}")

    fixed = _load_volume_xyz(fixed_path)
    baselines = _baseline_rows(base_run_dir)
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "run_kind": "v2_refine_from_downsample",
        "base_run_dir": str(base_run_dir),
        "out_dir": str(out_dir),
        "binary": str(binary),
        "case_ids": selected_case_ids,
        "composition": "refined = residual @ coarse",
        "rows": rows,
    }

    for case_id in selected_case_ids:
        case = cases[case_id]
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        coarse_path = base_run_dir / "full_downsample" / case_id / "transform.csv"
        expected_path = base_run_dir / "center_cube" / case_id / "expected_reg_transform.csv"
        moving_path = base_run_dir / "center_cube" / case_id / "moving.nii"
        if not coarse_path.exists():
            raise RuntimeError(f"coarse transform not found: {coarse_path}")
        if not expected_path.exists():
            raise RuntimeError(f"expected transform not found: {expected_path}")
        if not moving_path.exists():
            raise RuntimeError(f"moving center cube not found: {moving_path}")

        coarse = as_homogeneous_4x4(read_transform_csv(coarse_path))
        expected = read_transform_csv(expected_path)
        write_transform_csv(case_dir / "coarse_transform.csv", coarse[:3, :])
        write_transform_csv(case_dir / "expected_reg_transform.csv", expected)

        warp_start = time.perf_counter()
        moving = _load_volume_xyz(moving_path)
        fixed_to_moving = np.linalg.inv(coarse)
        write_transform_csv(case_dir / "fixed_to_moving_prewarp_transform.csv", fixed_to_moving[:3, :])
        prewarped = warp_reference_grid_to_source(
            moving,
            fixed_to_moving,
            output_shape=tuple(int(v) for v in fixed.shape),
        )
        warp_seconds = time.perf_counter() - warp_start
        prewarped_path = case_dir / "moving_prewarped.nii"
        save_nifti_v2(prewarped_path, prewarped, spacing=1.0)
        del moving
        del prewarped

        residual_transform_path = case_dir / "transform.csv"
        matches_path = case_dir / "matches.csv"
        registration_seconds = 0.0
        reg_exit_code = 0

        if not skip_registration:
            reg = RegSift3D(binary=binary)
            cmd = reg.build_command(
                fixed_path,
                prewarped_path,
                matches_path=matches_path,
                transform_path=residual_transform_path,
            )
            reg_env = dict(os.environ)
            prefix = ""
            if omp_num_threads is not None:
                reg_env["OMP_NUM_THREADS"] = str(omp_num_threads)
                prefix = f"OMP_NUM_THREADS={omp_num_threads} "
            start = time.perf_counter()
            with (case_dir / "run.log").open("w", encoding="utf-8") as log:
                log.write("$ " + prefix + " ".join(cmd) + "\n")
                log.flush()
                completed = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    env=reg_env,
                )
            reg_exit_code = completed.returncode
            registration_seconds = time.perf_counter() - start
        else:
            (case_dir / "run.log").write_text("registration skipped\n", encoding="utf-8")

        baseline_full = baselines.get(("full_downsample", case_id))
        baseline_center = baselines.get(("center_cube", case_id))
        row: dict[str, object] = {
            "case_id": case.case_id,
            "description": case.description,
            "coarse_transform_path": str(coarse_path),
            "expected_transform_path": str(expected_path),
            "moving_path": str(moving_path),
            "prewarped_path": str(prewarped_path),
            "warp_seconds": float(warp_seconds),
            "registration_seconds": float(registration_seconds),
            "reg_exit_code": int(reg_exit_code),
            "registration_succeeded": bool(
                (not skip_registration) and reg_exit_code == 0 and residual_transform_path.exists()
            ),
            "baseline_full_max_abs_transform_element_error": _float_from_row(
                baseline_full, "max_abs_transform_element_error"
            ),
            "baseline_full_translation_l2_error_voxels": _float_from_row(
                baseline_full, "translation_l2_error_voxels"
            ),
            "baseline_center_max_abs_transform_element_error": _float_from_row(
                baseline_center, "max_abs_transform_element_error"
            ),
            "baseline_center_translation_l2_error_voxels": _float_from_row(
                baseline_center, "translation_l2_error_voxels"
            ),
        }

        if row["registration_succeeded"]:
            residual_metrics = evaluate_run(case_dir)
            for key, value in residual_metrics.items():
                row[f"residual_{key}"] = value

            residual = as_homogeneous_4x4(read_transform_csv(residual_transform_path))
            refined = residual @ coarse
            refined_path = case_dir / "refined_transform.csv"
            write_transform_csv(refined_path, refined[:3, :])
            write_transform_csv(case_dir / "residual_transform.csv", residual[:3, :])

            matrix_stats = matrix_error_stats(expected, refined[:3, :])
            row.update({f"refined_{key}": value for key, value in matrix_stats.items()})
            center_err = row.get("baseline_center_max_abs_transform_element_error")
            full_err = row.get("baseline_full_max_abs_transform_element_error")
            if center_err is not None:
                row["refined_minus_center_max_abs_error"] = (
                    float(matrix_stats["max_abs_transform_element_error"]) - float(center_err)
                )
            if full_err is not None:
                row["refined_minus_full_max_abs_error"] = (
                    float(matrix_stats["max_abs_transform_element_error"]) - float(full_err)
                )
            write_matrix_element_errors_csv(
                case_dir / "matrix_error_elements.csv",
                expected,
                refined[:3, :],
            )
            (case_dir / "matrix_error.json").write_text(
                json.dumps(matrix_stats, indent=2),
                encoding="utf-8",
            )

        rows.append(row)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fields = sorted({key for row in rows for key in row.keys()})
    _write_csv(out_dir / "refinement_metrics.csv", rows, fields)
    _make_report(summary, out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
