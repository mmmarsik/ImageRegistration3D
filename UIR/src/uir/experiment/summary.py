"""Run summary contract: ``RunSummary`` TypedDict + ``RunPaths`` + loader.

This module names the shape of the per-run ``summary.json`` files that the
pipeline already writes and reads. It is descriptive, not enforced: nothing
here changes what writers emit or what readers consume. Fields below were
derived from real summaries under ``UIR/runs/`` covering all three run kinds:

- synthetic        (``cli/plot_single_case_report.py`` -> ``plots/summary.json``)
- real_pair        (``cli/summarize_real_pair_case.py`` -> ``summary.json``)
- resolution_pair  (``analysis/resolution_pair.py`` + resample -> ``summary.json``)

Because the three kinds share a file name but differ in keys, ``RunSummary``
is a single ``total=False`` TypedDict (every key optional) that is the union
of the keys observed across the three kinds. This mirrors how the readers
already treat the file: a flat ``dict[str, object]`` validated key-by-key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


# match_residual_stats keys, kept as a stable schema (see canonical decision
# in the module docstring section "Open question" below).
class MatchResidualStats(TypedDict, total=False):
    """Keys returned by ``analysis/transform_metrics.py:match_residual_stats``.

    Open question (resolved here, documentation only): the function's
    empty-input branch (``transform_metrics.py``) omits ``match_raw_l2_median``
    and ``match_raw_l2_max`` that the populated branch includes. The *canonical*
    schema chosen for this contract is the **full populated key set**, with
    ``None`` standing in for the empty case. This TypedDict therefore lists all
    keys as optional (``total=False``) so it accepts both the historical empty
    payload (which lacks the two raw-median/max keys) and the populated payload.
    No runtime code is changed; the function still emits its existing two
    branches. If a future task unifies the branches, padding the empty branch
    with ``match_raw_l2_median``/``match_raw_l2_max`` set to ``None`` makes the
    on-disk shape match this canonical schema without breaking old summaries.
    """

    match_residual_count: int
    match_raw_l2_mean: float | None
    match_raw_l2_median: float | None
    match_raw_l2_max: float | None
    match_residual_l2_mean: float | None
    match_residual_l2_median: float | None
    match_residual_l2_rms: float | None
    match_residual_l2_p95: float | None
    match_residual_l2_max: float | None
    match_residual_xyz_mean: list[float] | None


class RunSummary(TypedDict, total=False):
    """Union wire format of every per-run ``summary.json``.

    All keys are optional (``total=False``): each concrete run kind populates a
    subset, and the lenient loader tolerates any missing key. Keys are grouped
    below by origin; values match what the writers emit verbatim.
    """

    # --- discriminators (which kind of run this is) ---
    run_kind: str            # "synthetic" | "resolution_pair" (absent for real_pair)
    mode: str                # "real_pair" | "resample" (resolution_pair)

    # --- common registration outcome ---
    registration_succeeded: bool
    reg_exit_code: int
    run_dir: str
    matches_path: str
    transform_path: str
    match_count: int

    # --- synthetic-specific ---
    degradation: str         # "awgn" | "blur_awgn"
    transform_tag: str
    roi_size_xyz: list[int]
    blur_sigma_xy: float
    awgn_seed: int
    expected_transform_path: str
    estimated_transform_path: str
    transform_diff_path: str
    matrix_element_errors_path: str
    matrix_error_diagnostic_path: str
    match_residuals_path: str
    match_residual_diagnostic_path: str
    noise_reference_path: str
    noisy_path: str
    noise_mean: float
    noise_std: float
    noise_min: float
    noise_max: float
    clean_observation_min: float
    clean_observation_max: float
    noisy_observation_min: float
    noisy_observation_max: float
    awgn_variance: float
    requested_variance: float
    expected_noise_std: float
    noise_std_abs_error: float
    noise_variance_observed: float

    # --- real_pair-specific ---
    before_stack_dir: str
    after_stack_dir: str
    before_nifti: str
    after_nifti: str
    before_stack_shape_xyz: list[int]
    after_stack_shape_xyz: list[int]
    before_nifti_shape_xyz: list[int]
    after_nifti_shape_xyz: list[int]
    before_observation_range: list[float]
    after_observation_range: list[float]
    estimated_transform_rows: int
    estimated_transform_cols: int
    before_roi_start_xyz: list[int]
    after_roi_start_xyz: list[int]

    # --- resolution_pair-specific ---
    source_stack_dir: str
    ratio: int
    high_spacing: float
    low_spacing: float
    source_shape_xyz: list[int]
    low_res_shape_xyz: list[int]
    high_crop_shape_xyz: list[int]
    high_crop_start_xyz: list[int]
    low_res_whole_path: str
    high_res_crop_clean_path: str
    expected_transform_semantics: str
    mode_dir: str
    translation_error_physical_xyz: list[float]
    translation_l2_error_physical: float

    # --- matrix_error_stats fragment (synthetic + resolution_pair) ---
    linear_rms_error: float
    linear_mean_abs_error: float
    linear_max_abs_error: float
    linear_max_abs_error_component: str
    translation_l2_error_voxels: float
    translation_mean_abs_error_voxels: float
    translation_max_abs_error_voxels: float
    translation_max_abs_error_axis: str
    max_abs_transform_element_error: float
    max_abs_transform_element_component: str
    translation_error_xyz: list[float]
    expected_linear_det: float
    estimated_linear_det: float
    linear_det_abs_error: float

    # --- match_residual_stats fragment (synthetic + resolution_pair) ---
    match_residual_count: int
    match_raw_l2_mean: float | None
    match_raw_l2_median: float | None
    match_raw_l2_max: float | None
    match_residual_l2_mean: float | None
    match_residual_l2_median: float | None
    match_residual_l2_rms: float | None
    match_residual_l2_p95: float | None
    match_residual_l2_max: float | None
    match_residual_xyz_mean: list[float] | None


@dataclass(frozen=True)
class RunPaths:
    """Fixed file layout of a single run directory.

    Mirrors the repeated ``run_dir / "plots" / ...`` construction across the
    pipeline (see ``docs/audit/03_api_contracts.md`` artifact table). Pure path
    arithmetic; it does not touch the filesystem and creates nothing.
    """

    run_dir: Path

    @property
    def plots(self) -> Path:
        return self.run_dir / "plots"

    @property
    def matches(self) -> Path:
        return self.run_dir / "matches.csv"

    @property
    def transform(self) -> Path:
        return self.run_dir / "transform.csv"

    @property
    def summary(self) -> Path:
        return self.run_dir / "summary.json"

    @property
    def match_residuals(self) -> Path:
        return self.plots / "match_residuals.csv"

    @property
    def matrix_element_errors(self) -> Path:
        return self.plots / "matrix_element_errors.csv"


def load_run_summary(path: Path | str) -> RunSummary:
    """Load a ``summary.json`` leniently into a ``RunSummary``.

    Tolerant by design: any subset of keys is accepted and missing keys are
    simply absent (``RunSummary`` is ``total=False``). Unknown extra keys are
    preserved as-is. This does NOT validate or coerce values; it only parses
    the JSON object. Raises only if the file is missing or not a JSON object.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, got {type(data).__name__}"
        )
    # The cast is documentation-only; we do not narrow or drop keys.
    return data  # type: ignore[return-value]
