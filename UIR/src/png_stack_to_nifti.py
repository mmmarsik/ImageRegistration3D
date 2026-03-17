import argparse
from pathlib import Path
import re
import sys

import cv2
import nibabel as nib
import numpy as np


def extract_last_number(path: Path) -> int:
    m = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if m is None:
        raise ValueError(f"Cannot extract numeric suffix from filename: {path.name}")
    return int(m.group(1))


def read_png_as_gray_u16(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        gray = img
    else:
        raise RuntimeError(f"Unsupported image shape {img.shape} in file {path}")

    if gray.dtype == np.uint8:
        gray = gray.astype(np.uint16)
    elif gray.dtype == np.uint16:
        pass
    else:
        raise RuntimeError(f"Unsupported dtype {gray.dtype} in file {path}")

    return gray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PNG stack to NIfTI, optionally extracting an ROI."
    )
    parser.add_argument("input_dir")
    parser.add_argument("output_path")
    parser.add_argument("sx", nargs="?", type=float, default=1.0)
    parser.add_argument("sy", nargs="?", type=float, default=1.0)
    parser.add_argument("sz", nargs="?", type=float, default=1.0)
    parser.add_argument(
        "--roi-size",
        nargs=3,
        type=int,
        metavar=("X", "Y", "Z"),
        help="Extract an ROI of size X Y Z. If --roi-start is omitted, use the centered ROI.",
    )
    parser.add_argument(
        "--roi-start",
        nargs=3,
        type=int,
        metavar=("X0", "Y0", "Z0"),
        help="ROI origin in voxel coordinates for X Y Z.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    sx = args.sx
    sy = args.sy
    sz = args.sz

    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist or is not a directory: {input_dir}")

    pngs = sorted(input_dir.glob("*.png"), key=extract_last_number)
    if not pngs:
        raise RuntimeError(f"No PNG files found in directory: {input_dir}")

    z0 = 0
    roi_x = roi_y = roi_z = None
    if args.roi_size is not None:
        roi_x, roi_y, roi_z = args.roi_size
        if roi_x <= 0 or roi_y <= 0 or roi_z <= 0:
            raise RuntimeError("--roi-size values must be positive.")
        if args.roi_start is not None:
            x0, y0, z0 = args.roi_start
        else:
            if roi_z > len(pngs):
                raise RuntimeError(f"ROI depth {roi_z} exceeds number of slices {len(pngs)}.")
            z0 = (len(pngs) - roi_z) // 2
        if z0 < 0 or z0 + roi_z > len(pngs):
            raise RuntimeError(
                f"ROI Z range [{z0}, {z0 + roi_z}) is outside available slices [0, {len(pngs)})."
            )
        pngs = pngs[z0 : z0 + roi_z]
    else:
        x0 = y0 = 0

    slices = []
    ref_shape = None

    for i, p in enumerate(pngs):
        gray = read_png_as_gray_u16(p)

        if ref_shape is None:
            ref_shape = gray.shape
        elif gray.shape != ref_shape:
            raise RuntimeError(
                f"Slice shape mismatch: expected {ref_shape}, got {gray.shape} in file {p}"
            )

        if args.roi_size is not None:
            if args.roi_start is None and i == 0:
                x0 = (gray.shape[1] - roi_x) // 2
                y0 = (gray.shape[0] - roi_y) // 2
            if x0 < 0 or y0 < 0 or x0 + roi_x > gray.shape[1] or y0 + roi_y > gray.shape[0]:
                raise RuntimeError(
                    f"ROI XY range [{x0}, {x0 + roi_x}) x [{y0}, {y0 + roi_y}) "
                    f"is outside slice shape {gray.shape[::-1]}."
                )
            gray = gray[y0 : y0 + roi_y, x0 : x0 + roi_x]

        slices.append(gray)

    # Собираем в порядке z=0,1,2,... как (Z, Y, X)
    volume_zyx = np.stack(slices, axis=0)

    # была бага без перевода к (X, Y, Z) перед записью в NIfTI 
    volume_xyz = np.transpose(volume_zyx, (2, 1, 0))

    affine = np.array(
        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    nii = nib.Nifti1Image(volume_xyz, affine)
    nib.save(nii, str(output_path))

    print(f"Saved: {output_path}")
    print(f"Number of slices: {len(pngs)}")
    print(f"Input slice shape (Y, X): {ref_shape}")
    print(f"Saved volume shape (X, Y, Z): {volume_xyz.shape}")
    print(f"dtype: {volume_xyz.dtype}")
    print(f"spacing: ({sx}, {sy}, {sz})")
    print(f"First file: {pngs[0].name}")
    print(f"Last file:  {pngs[-1].name}")
    if args.roi_size is not None:
        print(f"ROI start (X, Y, Z): ({x0}, {y0}, {z0})")
        print(f"ROI size  (X, Y, Z): ({roi_x}, {roi_y}, {roi_z})")


if __name__ == "__main__":
    main()
