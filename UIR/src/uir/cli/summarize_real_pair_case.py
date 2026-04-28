from __future__ import annotations

import argparse
import json
from pathlib import Path

from uir.analysis.real_pair_summary import summarize_real_pair_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a compact summary for a real pairwise registration case."
    )
    parser.add_argument("run_dir", type=Path, help="Run directory for the case outputs.")
    parser.add_argument("--before-stack-dir", type=Path, required=True, help="Directory with the before PNG stack.")
    parser.add_argument("--after-stack-dir", type=Path, required=True, help="Directory with the after PNG stack.")
    parser.add_argument("--before-nifti", type=Path, required=True, help="Converted before NIfTI path.")
    parser.add_argument("--after-nifti", type=Path, required=True, help="Converted after NIfTI path.")
    parser.add_argument("--matches-path", type=Path, required=True, help="CSV path with SIFT3D matches.")
    parser.add_argument("--transform-path", type=Path, required=True, help="CSV path with estimated transform.")
    parser.add_argument("--reg-exit-code", type=int, default=0, help="Exit code returned by regSift3D.")
    parser.add_argument("--roi-size", nargs=3, type=int, metavar=("X", "Y", "Z"))
    parser.add_argument("--before-roi-start", nargs=3, type=int, metavar=("X0", "Y0", "Z0"))
    parser.add_argument("--after-roi-start", nargs=3, type=int, metavar=("X0", "Y0", "Z0"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_real_pair_case(
        run_dir=args.run_dir,
        before_stack_dir=args.before_stack_dir,
        after_stack_dir=args.after_stack_dir,
        before_nifti=args.before_nifti,
        after_nifti=args.after_nifti,
        matches_path=args.matches_path,
        transform_path=args.transform_path,
        reg_exit_code=args.reg_exit_code,
        roi_size_xyz=tuple(args.roi_size) if args.roi_size is not None else None,
        before_roi_start_xyz=tuple(args.before_roi_start) if args.before_roi_start is not None else None,
        after_roi_start_xyz=tuple(args.after_roi_start) if args.after_roi_start is not None else None,
    )

    summary_path = args.run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Run dir: {args.run_dir}")
    print(f"Registration succeeded: {summary['registration_succeeded']}")
    print(f"Match count: {summary['match_count']}")
    if "estimated_linear_det" in summary:
        print(f"Estimated linear det: {summary['estimated_linear_det']:.6f}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
