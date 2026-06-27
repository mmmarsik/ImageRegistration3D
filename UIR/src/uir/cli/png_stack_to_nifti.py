from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from uir.io.nifti_volume import make_affine, save_nifti
from uir.io.png_stack import VOLUME_DTYPES, load_png_stack_volume
from uir.observation.model import ObservationModel, read_observation_model


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
    parser.add_argument(
        "--observation-model-like",
        type=Path,
        help="Use the observation model sidecar associated with this reference volume.",
    )
    parser.add_argument("--observation-min", type=float, help="Override observation model minimum.")
    parser.add_argument("--observation-max", type=float, help="Override observation model maximum.")
    parser.add_argument(
        "--dtype",
        choices=sorted(VOLUME_DTYPES),
        default="float32",
        help="NIfTI voxel dtype to write. Use uint8 for native 8-bit real stacks to reduce file and read memory.",
    )
    return parser.parse_args()


def resolve_observation_model_override(args: argparse.Namespace) -> ObservationModel | None:
    if args.observation_model_like is not None:
        if args.observation_min is not None or args.observation_max is not None:
            raise RuntimeError("Use either --observation-model-like or --observation-min/--observation-max, not both.")
        return read_observation_model(args.observation_model_like)

    if args.observation_min is None and args.observation_max is None:
        return None

    if args.observation_min is None or args.observation_max is None:
        raise RuntimeError("Both --observation-min and --observation-max are required when overriding observation model.")

    return ObservationModel(args.observation_min, args.observation_max)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    observation_model_override = resolve_observation_model_override(args)
    stack = load_png_stack_volume(
        input_dir,
        roi_size_xyz=tuple(args.roi_size) if args.roi_size is not None else None,
        roi_start_xyz=tuple(args.roi_start) if args.roi_start is not None else None,
        observation_model_override=observation_model_override,
        volume_dtype=VOLUME_DTYPES[args.dtype],
    )
    affine = make_affine(args.sx, args.sy, args.sz)
    save_nifti(output_path, stack.volume_xyz, affine, observation_model=stack.observation_model, dtype=VOLUME_DTYPES[args.dtype])

    print(f"Saved: {output_path}")
    print(f"Number of slices: {stack.volume_xyz.shape[2]}")
    print(f"Input slice shape (Y, X): {stack.input_shape_yx}")
    print(f"Saved volume shape (X, Y, Z): {stack.volume_xyz.shape}")
    print(f"dtype: {np.dtype(VOLUME_DTYPES[args.dtype]).name}")
    print(f"spacing: ({args.sx}, {args.sy}, {args.sz})")
    print(f"Observation range: [{stack.observation_model.min_value:.1f}, {stack.observation_model.max_value:.1f}]")
    print(f"Decoded PNG dtype: {stack.decoded_dtype}")
    print(f"First file: {stack.first_file}")
    print(f"Last file:  {stack.last_file}")
    if stack.roi_start_xyz is not None and args.roi_size is not None:
        x0, y0, z0 = stack.roi_start_xyz
        roi_x, roi_y, roi_z = args.roi_size
        print(f"ROI start (X, Y, Z): ({x0}, {y0}, {z0})")
        print(f"ROI size  (X, Y, Z): ({roi_x}, {roi_y}, {roi_z})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
