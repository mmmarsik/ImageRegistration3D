from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from uir.v2.experiment import default_regsift3d_binary
from uir.v2.refinement_pipeline import DEFAULT_REFINE_CASE_IDS, run_refinement_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m uir.cli.v2_refine_from_downsample",
        description=(
            "Use full-downsample transforms as coarse approximations, prewarp center-cube "
            "moving volumes, run residual regSift3D, and compare the refined matrices."
        ),
    )
    parser.add_argument(
        "--base-run-dir",
        type=Path,
        default=Path("UIR/runs/v2/affine_real_big_500"),
        help="Existing v2 affine run with full_downsample and center_cube outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("UIR/runs/v2/refine_from_downsample_500"),
    )
    parser.add_argument("--binary", type=Path, default=None, help="regSift3D binary.")
    parser.add_argument("--omp-num-threads", type=int, default=None)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=DEFAULT_REFINE_CASE_IDS,
        help="Case ids to run.",
    )
    parser.add_argument(
        "--skip-registration",
        action="store_true",
        help="Only write prewarped NIfTI volumes; useful for smoke-testing the warp step.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    binary = args.binary if args.binary is not None else default_regsift3d_binary()
    if args.omp_num_threads is None:
        raw_threads = os.environ.get("OMP_NUM_THREADS")
        omp_num_threads = int(raw_threads) if raw_threads else None
    else:
        omp_num_threads = args.omp_num_threads

    if not binary.exists():
        raise SystemExit(f"regSift3D binary not found: {binary}")
    if not args.base_run_dir.exists():
        raise SystemExit(f"base run dir not found: {args.base_run_dir}")

    summary = run_refinement_pipeline(
        base_run_dir=args.base_run_dir,
        out_dir=args.out_dir,
        binary=binary,
        case_ids=list(args.cases),
        omp_num_threads=omp_num_threads,
        skip_registration=args.skip_registration,
    )
    print(
        json.dumps(
            {
                "out_dir": summary["out_dir"],
                "case_count": len(summary["case_ids"]),
                "report": str(Path(str(summary["out_dir"])) / "REPORT.md"),
                "summary": str(Path(str(summary["out_dir"])) / "summary.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
