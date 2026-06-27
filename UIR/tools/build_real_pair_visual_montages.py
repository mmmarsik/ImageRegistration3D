from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "UIR" / "src"))

from uir.reporting.real_pair_stacks import (  # noqa: E402
    _as_homogeneous_4x4,
    _gray_rgb_slice,
    _resample_after_slice_to_before_grid_with_mask,
    _volume_display_window,
)
from uir.transforms.io import read_transform_csv  # noqa: E402


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = ["Arial Bold.ttf", "Arial.ttf"] if bold else ["Arial.ttf", "Helvetica.ttc"]
    for name in names:
        candidate = Path("/System/Library/Fonts/Supplemental") / name
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def to_rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def registered_after_slice_rgb(
    after_xyz: np.ndarray,
    after_window: tuple[float, float],
    before_to_after: np.ndarray,
    z_before: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> Image.Image:
    registered_after, valid_mask = _resample_after_slice_to_before_grid_with_mask(
        after_xyz,
        before_to_after,
        z_before,
        x_grid,
        y_grid,
    )
    # Mark regions not covered by the transformed after volume as white/no data.
    registered_after = np.where(valid_mask, registered_after, after_window[1])
    return Image.fromarray(_gray_rgb_slice(registered_after, after_window))


def resize_into_cell(panel: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    ratio = min(cell_w / panel.width, cell_h / panel.height)
    new_size = (max(1, int(panel.width * ratio)), max(1, int(panel.height * ratio)))
    return panel.resize(new_size, Image.Resampling.LANCZOS)


def build_montage(
    *,
    run_dir: Path,
    slice_indices: list[int],
    out_path: Path,
) -> None:
    plots_dir = run_dir / "plots"
    before_dir = plots_dir / "matched_keypoints_before_png_stack"
    diff_dir = plots_dir / "signed_diff_png_stack"

    before_img = nib.load(str(run_dir / "before.nii"))
    after_img = nib.load(str(run_dir / "after.nii"))
    before_xyz = np.asarray(before_img.get_fdata(dtype=np.float32), dtype=np.float32)
    after_xyz = np.asarray(after_img.get_fdata(dtype=np.float32), dtype=np.float32)
    transform = read_transform_csv(run_dir / "transform.csv")
    before_to_after = np.linalg.inv(_as_homogeneous_4x4(transform))
    after_window = _volume_display_window(after_img, after_xyz)

    x_grid, y_grid = np.meshgrid(
        np.arange(before_xyz.shape[0], dtype=np.float64),
        np.arange(before_xyz.shape[1], dtype=np.float64),
        indexing="ij",
    )

    rows: list[tuple[int, Image.Image, Image.Image, Image.Image]] = []
    for z in slice_indices:
        if not (0 <= z < before_xyz.shape[2]):
            raise RuntimeError(f"Slice z={z} is outside before volume depth {before_xyz.shape[2]}")
        before_panel = to_rgb(Image.open(before_dir / f"slice_{z:04d}.png"))
        registered_after_panel = registered_after_slice_rgb(
            after_xyz,
            after_window,
            before_to_after,
            z,
            x_grid,
            y_grid,
        )
        diff_panel = to_rgb(Image.open(diff_dir / f"slice_{z:04d}.png"))
        rows.append((z, before_panel, registered_after_panel, diff_panel))

    target_h = 240
    cell_w = max(
        max(panel.width * target_h // panel.height for panel in (before, registered_after, diff))
        for _, before, registered_after, diff in rows
    )
    cell_h = target_h
    gap = 14
    label_h = 28
    title_h = 30
    canvas_w = 3 * cell_w + 4 * gap
    canvas_h = title_h + len(rows) * (cell_h + label_h + gap) + gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(15)
    z_font = load_font(20, bold=True)
    col_x = [gap + i * (cell_w + gap) for i in range(3)]
    col_labels = (
        "(1) до + точки",
        "(2) после -> до",
        "(3) знаковая разность",
    )
    for x, label in zip(col_x, col_labels):
        draw.text((x, 5), label, fill=(50, 50, 50), font=label_font)

    y = title_h
    for z, before_panel, registered_after_panel, diff_panel in rows:
        draw.text((gap, y), f"z = {z}", fill=(20, 20, 20), font=z_font)
        y += label_h
        for x, panel in zip(col_x, (before_panel, registered_after_panel, diff_panel)):
            resized = resize_into_cell(panel, cell_w, cell_h)
            canvas.paste(
                resized,
                (
                    x + (cell_w - resized.width) // 2,
                    y + (cell_h - resized.height) // 2,
                ),
            )
        y += cell_h + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved {out_path} size={canvas.size}")


def main() -> int:
    summary_dir = ROOT / "UIR" / "runs" / "real_pair" / "summary"
    build_montage(
        run_dir=ROOT / "UIR" / "runs" / "real_pair" / "real3d_pair_peak020" / "full_volume",
        slice_indices=[240, 355, 470],
        out_path=summary_dir / "real1_peak020_visual_montage.png",
    )
    build_montage(
        run_dir=ROOT
        / "UIR"
        / "runs"
        / "real_pair"
        / "real3d_pair_another_uint8_peak020"
        / "roi650",
        slice_indices=[200, 325, 450],
        out_path=summary_dir / "real2_roi650_peak020_visual_montage.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
