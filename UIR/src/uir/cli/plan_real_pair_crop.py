from __future__ import annotations

import argparse
from pathlib import Path

from uir.analysis.real_pair_crop import plan_common_centered_roi_from_stacks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan a common centered ROI for a pair of PNG stacks."
    )
    parser.add_argument("before_stack_dir", type=Path)
    parser.add_argument("after_stack_dir", type=Path)
    parser.add_argument("--roi-size", nargs=3, type=int, metavar=("X", "Y", "Z"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = plan_common_centered_roi_from_stacks(
        args.before_stack_dir,
        args.after_stack_dir,
        requested_size_xyz=tuple(args.roi_size) if args.roi_size is not None else None,
    )
    roi_x, roi_y, roi_z = plan["roi_size_xyz"]
    before_x, before_y, before_z = plan["before_roi_start_xyz"]
    after_x, after_y, after_z = plan["after_roi_start_xyz"]
    print(roi_x, roi_y, roi_z, before_x, before_y, before_z, after_x, after_y, after_z)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
