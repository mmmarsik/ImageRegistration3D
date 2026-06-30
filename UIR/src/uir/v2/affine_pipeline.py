from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from uir.core.io.png_stack import list_png_paths
from uir.core.transforms.io import read_transform_csv
from uir.core.transforms.metrics import matrix_error_stats, write_matrix_element_errors_csv
from uir.core.transforms.matrix import as_homogeneous_4x4
from uir.registration.regsift3d import RegSift3D
from uir.v2.downsample import downsample_png_stack
from uir.v2.io import save_nifti_v2
from uir.v2.metrics import evaluate_run


@dataclass(frozen=True)
class AffineCase:
    case_id: str
    description: str
    rx_deg: float = 0.0
    ry_deg: float = 0.0
    rz_deg: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    sz: float = 1.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0


@dataclass(frozen=True)
class WorkMode:
    mode_id: str
    description: str
    high_crop_size: int
    downsample_ratio: int
    target_size: int


def default_cases() -> list[AffineCase]:
    return [
        AffineCase("t01_translation_small", "translation only, small", tx=12.0, ty=-8.0, tz=5.0),
        AffineCase("t02_translation_medium", "translation only, medium", tx=28.0, ty=-18.0, tz=12.0),
        AffineCase("t03_rotation_z_small", "rotation only around z", rz_deg=2.0),
        AffineCase("t04_rotation_xyz_small", "small xyz rotations", rx_deg=1.5, ry_deg=-2.0, rz_deg=3.0),
        AffineCase("t05_rotation_xyz_medium", "medium xyz rotations", rx_deg=4.0, ry_deg=-5.0, rz_deg=7.0),
        AffineCase("t06_scale_up_small", "uniform scale up", sx=1.03, sy=1.03, sz=1.03),
        AffineCase("t07_scale_down_small", "uniform scale down", sx=0.97, sy=0.97, sz=0.97),
        AffineCase("t08_scale_anisotropic", "anisotropic scale", sx=1.04, sy=0.96, sz=1.02),
        AffineCase("t09_rotation_translation", "rotation plus translation", ry_deg=-3.0, rz_deg=5.0, tx=20.0, ty=-12.0, tz=9.0),
        AffineCase(
            "t10_combo_hard",
            "rotation plus anisotropic scale plus translation",
            rx_deg=6.0,
            ry_deg=-5.0,
            rz_deg=9.0,
            sx=1.05,
            sy=0.95,
            sz=1.03,
            tx=24.0,
            ty=-18.0,
            tz=13.0,
        ),
    ]


def default_work_modes(target_size: int) -> list[WorkMode]:
    return [
        WorkMode(
            "full_downsample",
            "central 2x larger source cube, block-averaged to target size",
            high_crop_size=target_size * 2,
            downsample_ratio=2,
            target_size=target_size,
        ),
        WorkMode(
            "center_cube",
            "central source cube at native resolution",
            high_crop_size=target_size,
            downsample_ratio=1,
            target_size=target_size,
        ),
    ]


def inspect_stack(input_dir: Path) -> dict[str, object]:
    pngs = list_png_paths(input_dir)
    with Image.open(pngs[0]) as im:
        width, height = im.size
        mode = im.mode
    return {
        "input_dir": str(input_dir),
        "slices": len(pngs),
        "width": int(width),
        "height": int(height),
        "mode": mode,
        "voxel_count": int(width * height * len(pngs)),
        "first_file": str(pngs[0]),
        "last_file": str(pngs[-1]),
    }


def validate_crop(stack: dict[str, object], crop_size: int) -> None:
    width = int(stack["width"])
    height = int(stack["height"])
    depth = int(stack["slices"])
    if crop_size > min(width, height, depth):
        raise RuntimeError(
            f"crop_size={crop_size} does not fit stack shape xyz={(width, height, depth)}"
        )


def crop_start(stack: dict[str, object], crop_size: int) -> tuple[int, int, int]:
    width = int(stack["width"])
    height = int(stack["height"])
    depth = int(stack["slices"])
    return (
        (width - crop_size) // 2,
        (height - crop_size) // 2,
        (depth - crop_size) // 2,
    )


def _read_crop(path: Path, x0: int, y0: int, crop_size: int) -> np.ndarray:
    with Image.open(path) as im:
        gray = im.convert("L")
        crop = gray.crop((x0, y0, x0 + crop_size, y0 + crop_size))
        return np.asarray(crop, dtype=np.float32)


def load_center_cube(input_dir: Path, crop_size: int) -> tuple[np.ndarray, dict[str, object]]:
    stack = inspect_stack(input_dir)
    validate_crop(stack, crop_size)
    x0, y0, z0 = crop_start(stack, crop_size)
    pngs = list_png_paths(input_dir)

    volume_zyx = np.empty((crop_size, crop_size, crop_size), dtype=np.float32)
    for out_z, src_z in enumerate(range(z0, z0 + crop_size)):
        volume_zyx[out_z] = _read_crop(pngs[src_z], x0, y0, crop_size)

    meta = {
        "source": "center_cube",
        "crop_start_xyz": [int(x0), int(y0), int(z0)],
        "high_crop_size": int(crop_size),
        "downsample_ratio": 1,
        "shape_xyz": [int(crop_size), int(crop_size), int(crop_size)],
    }
    return np.transpose(volume_zyx, (2, 1, 0)), meta


def load_center_block_downsample(
    input_dir: Path,
    *,
    high_crop_size: int,
    ratio: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if high_crop_size % ratio != 0:
        raise ValueError("high_crop_size must be divisible by ratio")

    stack = inspect_stack(input_dir)
    validate_crop(stack, high_crop_size)
    x0, y0, z0 = crop_start(stack, high_crop_size)
    pngs = list_png_paths(input_dir)
    low_size = high_crop_size // ratio
    volume_zyx = np.empty((low_size, low_size, low_size), dtype=np.float32)

    inv_block = np.float32(1.0 / float(ratio**3))
    for out_z, src_z0 in enumerate(range(z0, z0 + high_crop_size, ratio)):
        group_sum = np.zeros((high_crop_size, high_crop_size), dtype=np.float32)
        for src_z in range(src_z0, src_z0 + ratio):
            group_sum += _read_crop(pngs[src_z], x0, y0, high_crop_size)
        block_sum = group_sum.reshape(low_size, ratio, low_size, ratio).sum(axis=(1, 3))
        volume_zyx[out_z] = block_sum * inv_block

    meta = {
        "source": "center_block_downsample",
        "crop_start_xyz": [int(x0), int(y0), int(z0)],
        "high_crop_size": int(high_crop_size),
        "downsample_ratio": int(ratio),
        "shape_xyz": [int(low_size), int(low_size), int(low_size)],
    }
    return np.transpose(volume_zyx, (2, 1, 0)), meta


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_png_stack(volume_xyz: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    volume = np.clip(np.rint(volume_xyz), 0.0, 65535.0).astype(np.uint16)
    for z in range(volume.shape[2]):
        Image.fromarray(volume[:, :, z].T, mode="I;16").save(out_dir / f"slice_{z:04d}.png")


def case_env(case: AffineCase, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env.update(
        {
            "UIR_ROT_X_DEG": str(case.rx_deg),
            "UIR_ROT_Y_DEG": str(case.ry_deg),
            "UIR_ROT_Z_DEG": str(case.rz_deg),
            "UIR_SCALE_X": str(case.sx),
            "UIR_SCALE_Y": str(case.sy),
            "UIR_SCALE_Z": str(case.sz),
            "UIR_SH_XY": "0",
            "UIR_SH_XZ": "0",
            "UIR_SH_YX": "0",
            "UIR_SH_YZ": "0",
            "UIR_SH_ZX": "0",
            "UIR_SH_ZY": "0",
            "UIR_TX": str(case.tx),
            "UIR_TY": str(case.ty),
            "UIR_TZ": str(case.tz),
        }
    )
    return env


def run_cpp_generator(
    *,
    uir_affine_binary: Path,
    fixed_png_dir: Path,
    moving_png_dir: Path,
    artifact_dir: Path,
    case: AffineCase,
    log_path: Path,
    timing_json_path: Path,
    omp_num_threads: int | None,
) -> tuple[int, float]:
    env = case_env(case)
    if omp_num_threads is not None:
        env["OMP_NUM_THREADS"] = str(omp_num_threads)
    cmd = [
        str(uir_affine_binary),
        str(moving_png_dir),
        str(artifact_dir),
        "--input-stack",
        str(fixed_png_dir),
        "--timing-json",
        str(timing_json_path),
    ]
    start = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        prefix = "" if omp_num_threads is None else f"OMP_NUM_THREADS={omp_num_threads} "
        log.write("$ " + prefix + " ".join(cmd) + "\n")
        log.flush()
        completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False, env=env)
    return completed.returncode, time.perf_counter() - start


def _case_parameters(case: AffineCase) -> dict[str, object]:
    return {
        "rx_deg": case.rx_deg,
        "ry_deg": case.ry_deg,
        "rz_deg": case.rz_deg,
        "sx": case.sx,
        "sy": case.sy,
        "sz": case.sz,
        "tx": case.tx,
        "ty": case.ty,
        "tz": case.tz,
    }


def _make_report(summary: dict[str, object], out_dir: Path) -> None:
    rows = list(summary["rows"])
    succeeded = [row for row in rows if row["registration_succeeded"]]
    failed = [row for row in rows if not row["registration_succeeded"]]
    comparisons = list(summary["mode_comparisons"])

    lines: list[str] = [
        "# v2 affine pipeline report",
        "",
        "## Input",
        "",
        f"- source stack: `{summary['input_stack']['input_dir']}`",
        f"- stack shape xyz: `{summary['input_stack']['width']} x {summary['input_stack']['height']} x {summary['input_stack']['slices']}`",
        f"- target working volume shape: `{summary['target_size']}^3`",
        f"- cases planned: `{summary['case_count']}`",
        f"- registration binary: `{summary['binary']}`",
        "",
        "## Pipeline",
        "",
        f"1. `full_downsample`: central `{int(summary['target_size']) * 2}^3` source crop is block-averaged by 2 to `{summary['target_size']}^3`.",
        f"2. `center_cube`: central `{summary['target_size']}^3` source crop is used without downsampling.",
        "3. For each affine case, `moving` is generated from `fixed` by the existing C++ `uir_affine` generator.",
        "4. regSift3D registers `fixed` as reference and `moving` as source.",
        "5. The estimated matrix is compared with the known expected matrix and with the other mode.",
        "",
        "## Results",
        "",
        f"- successful registrations: `{len(succeeded)} / {len(rows)}`",
        f"- failed registrations: `{len(failed)}`",
        "",
    ]

    if failed:
        lines.extend(["### Failed Runs", ""])
        for row in failed:
            lines.append(f"- `{row['mode']}/{row['case_id']}` exit={row['reg_exit_code']}")
        lines.append("")

    lines.extend(
        [
            "### Matrix Error Against Expected",
            "",
            "| mode | case | match_count | p95 match residual | max matrix abs error | translation L2 error | time, s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {mode} | {case_id} | {match_count} | {p95} | {max_abs} | {translation_l2} | {seconds:.2f} |".format(
                mode=row["mode"],
                case_id=row["case_id"],
                match_count=row.get("match_count", ""),
                p95=_fmt(row.get("match_residual_l2_p95")),
                max_abs=_fmt(row.get("max_abs_transform_element_error")),
                translation_l2=_fmt(row.get("translation_l2_error_voxels")),
                seconds=float(row.get("registration_seconds", 0.0)),
            )
        )

    lines.extend(
        [
            "",
            "### Full Downsample vs Center Cube",
            "",
            "| case | matrix max abs diff | translation L2 diff | note |",
            "|---|---:|---:|---|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {case_id} | {max_abs} | {translation_l2} | {note} |".format(
                case_id=row["case_id"],
                max_abs=_fmt(row.get("max_abs_transform_element_error")),
                translation_l2=_fmt(row.get("translation_l2_error_voxels")),
                note=row.get("note", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Important Notes",
            "",
            "- `match_residual_l2_p95` is computed on matched keypoints, not on all voxels.",
            "- Matrix errors are in the local `500^3` working coordinate system.",
            "- This pipeline deliberately avoids loading the full original real stack into RAM; only the two working `500^3` volumes are materialized.",
            "- A full-downsample matrix and a center-cube matrix can differ because they observe different physical context and keypoint distributions. There is no automatic correction unless the difference is stable across cases or predictable from crop/resolution geometry.",
            "",
            "## Reproducibility",
            "",
            "Run:",
            "",
            "```bash",
            "./UIR/run_v2_affine_pipeline.sh",
            "```",
            "",
            "Machine-readable outputs:",
            "",
            "- `summary.json`",
            "- `case_metrics.csv`",
            "- `mode_matrix_comparison.csv`",
            "- per-case `expected_reg_transform.csv`, `transform.csv`, `matrix_error.json`, `run.log`",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: object) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6g}"


def run_affine_pipeline(
    *,
    input_stack: Path,
    out_dir: Path,
    binary: Path,
    uir_affine_binary: Path,
    target_size: int = 500,
    max_cases: int | None = None,
    omp_num_threads: int | None = None,
) -> dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_stack = Path(input_stack)
    binary = Path(binary)
    uir_affine_binary = Path(uir_affine_binary)

    stack = inspect_stack(input_stack)
    modes = default_work_modes(target_size)
    cases = default_cases()
    if max_cases is not None:
        cases = cases[:max_cases]

    summary: dict[str, object] = {
        "run_kind": "v2_affine_pipeline",
        "input_stack": stack,
        "out_dir": str(out_dir),
        "binary": str(binary),
        "uir_affine_binary": str(uir_affine_binary),
        "target_size": int(target_size),
        "case_count": len(cases),
        "modes": [],
        "cases": [
            {"case_id": case.case_id, "description": case.description, **_case_parameters(case)}
            for case in cases
        ],
        "rows": [],
        "mode_comparisons": [],
    }

    rows: list[dict[str, object]] = []
    estimated_by_case: dict[str, dict[str, np.ndarray]] = {}

    for mode in modes:
        mode_dir = out_dir / mode.mode_id
        mode_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        if mode.downsample_ratio == 1:
            fixed, prep_meta = load_center_cube(input_stack, mode.high_crop_size)
        else:
            fixed, prep_meta = load_center_block_downsample(
                input_stack,
                high_crop_size=mode.high_crop_size,
                ratio=mode.downsample_ratio,
            )
        prep_seconds = time.perf_counter() - t0
        fixed_path = mode_dir / "fixed.nii"
        save_nifti_v2(fixed_path, fixed, spacing=float(mode.downsample_ratio))
        fixed_png_dir = mode_dir / "fixed_png_stack"
        t_png = time.perf_counter()
        write_png_stack(fixed, fixed_png_dir)
        fixed_png_seconds = time.perf_counter() - t_png

        mode_summary = {
            "mode_id": mode.mode_id,
            "description": mode.description,
            "fixed_path": str(fixed_path),
            "fixed_png_stack_dir": str(fixed_png_dir),
            "preparation_seconds": float(prep_seconds),
            "fixed_png_write_seconds": float(fixed_png_seconds),
            **prep_meta,
        }
        summary["modes"].append(mode_summary)
        (mode_dir / "mode.json").write_text(json.dumps(mode_summary, indent=2), encoding="utf-8")

        for case in cases:
            case_dir = mode_dir / case.case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            moving_png_dir = case_dir / "moving_png_stack"
            generator_exit_code, transform_seconds = run_cpp_generator(
                uir_affine_binary=uir_affine_binary,
                fixed_png_dir=fixed_png_dir,
                moving_png_dir=moving_png_dir,
                artifact_dir=case_dir,
                case=case,
                log_path=case_dir / "uir_affine.log",
                timing_json_path=case_dir / "uir_affine_timing.json",
                omp_num_threads=omp_num_threads,
            )
            expected_path = case_dir / "expected_reg_transform.csv"

            moving_path = case_dir / "moving.nii"
            if generator_exit_code == 0:
                moving = downsample_png_stack(moving_png_dir, 1.0)
                save_nifti_v2(moving_path, moving, spacing=float(mode.downsample_ratio))
                del moving

            matches_path = case_dir / "matches.csv"
            transform_path = case_dir / "transform.csv"
            if generator_exit_code == 0:
                reg = RegSift3D(binary=binary)
                cmd = reg.build_command(
                    fixed_path,
                    moving_path,
                    matches_path=matches_path,
                    transform_path=transform_path,
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
                exit_code = completed.returncode
                registration_seconds = time.perf_counter() - start
            else:
                exit_code = -1
                registration_seconds = 0.0

            row: dict[str, object] = {
                "mode": mode.mode_id,
                "case_id": case.case_id,
                "description": case.description,
                **_case_parameters(case),
                "fixed_path": str(fixed_path),
                "moving_path": str(moving_path),
                "expected_transform_path": str(expected_path),
                "estimated_transform_path": str(transform_path),
                "matches_path": str(matches_path),
                "transform_generation_seconds": float(transform_seconds),
                "generator_exit_code": int(generator_exit_code),
                "registration_seconds": float(registration_seconds),
                "reg_exit_code": int(exit_code),
                "registration_succeeded": bool(
                    generator_exit_code == 0 and exit_code == 0 and transform_path.exists()
                ),
            }

            if row["registration_succeeded"]:
                metrics = evaluate_run(case_dir)
                row.update(metrics)
                estimated = read_transform_csv(transform_path)
                estimated_by_case.setdefault(case.case_id, {})[mode.mode_id] = as_homogeneous_4x4(estimated)
                expected = read_transform_csv(expected_path)
                expected_for_compare = expected if estimated.shape == (3, 4) else as_homogeneous_4x4(expected)
                matrix_stats = matrix_error_stats(expected_for_compare, estimated)
                row.update(matrix_stats)
                write_matrix_element_errors_csv(
                    case_dir / "matrix_error_elements.csv",
                    expected_for_compare,
                    estimated,
                )
                (case_dir / "matrix_error.json").write_text(
                    json.dumps(matrix_stats, indent=2),
                    encoding="utf-8",
                )
            rows.append(row)
            summary["rows"] = rows
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        del fixed

    comparison_rows: list[dict[str, object]] = []
    for case in cases:
        by_mode = estimated_by_case.get(case.case_id, {})
        full = by_mode.get("full_downsample")
        cube = by_mode.get("center_cube")
        if full is None or cube is None:
            comparison_rows.append(
                {
                    "case_id": case.case_id,
                    "note": "missing transform in one or both modes",
                }
            )
            continue
        stats = matrix_error_stats(full[:3, :], cube[:3, :])
        comparison_rows.append(
            {
                "case_id": case.case_id,
                "note": "center_cube_estimated_minus_full_downsample_estimated",
                **stats,
            }
        )

    summary["mode_comparisons"] = comparison_rows
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    case_fields = sorted({key for row in rows for key in row.keys()})
    _write_csv(out_dir / "case_metrics.csv", rows, case_fields)
    comparison_fields = sorted({key for row in comparison_rows for key in row.keys()})
    _write_csv(out_dir / "mode_matrix_comparison.csv", comparison_rows, comparison_fields)
    _make_report(summary, out_dir)
    return summary
