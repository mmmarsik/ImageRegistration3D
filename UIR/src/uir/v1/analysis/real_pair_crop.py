from __future__ import annotations

from pathlib import Path

from uir.core.io.png_stack import inspect_png_stack


def centered_roi_start(full_shape_xyz: tuple[int, int, int], roi_size_xyz: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple((full - roi) // 2 for full, roi in zip(full_shape_xyz, roi_size_xyz))


def plan_common_centered_roi(
    before_shape_xyz: tuple[int, int, int],
    after_shape_xyz: tuple[int, int, int],
    requested_size_xyz: tuple[int, int, int] | None = None,
) -> dict[str, tuple[int, int, int]]:
    common_shape_xyz = tuple(min(before, after) for before, after in zip(before_shape_xyz, after_shape_xyz))

    if requested_size_xyz is None:
        roi_size_xyz = common_shape_xyz
    else:
        roi_size_xyz = requested_size_xyz
        for axis, requested, common in zip("XYZ", roi_size_xyz, common_shape_xyz):
            if requested <= 0:
                raise RuntimeError(f"Requested ROI {axis} must be positive, got {requested}.")
            if requested > common:
                raise RuntimeError(
                    f"Requested ROI {axis}={requested} exceeds common paired extent {common}."
                )

    return {
        "roi_size_xyz": roi_size_xyz,
        "before_roi_start_xyz": centered_roi_start(before_shape_xyz, roi_size_xyz),
        "after_roi_start_xyz": centered_roi_start(after_shape_xyz, roi_size_xyz),
    }


def plan_common_centered_roi_from_stacks(
    before_stack_dir: Path,
    after_stack_dir: Path,
    requested_size_xyz: tuple[int, int, int] | None = None,
) -> dict[str, tuple[int, int, int]]:
    _, before_shape_xyz, _ = inspect_png_stack(before_stack_dir)
    _, after_shape_xyz, _ = inspect_png_stack(after_stack_dir)
    plan = plan_common_centered_roi(before_shape_xyz, after_shape_xyz, requested_size_xyz)
    plan["before_shape_xyz"] = before_shape_xyz
    plan["after_shape_xyz"] = after_shape_xyz
    return plan
