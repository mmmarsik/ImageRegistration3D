from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from uir.v2.affine_pipeline import run_affine_pipeline
from uir.v2.experiment import default_regsift3d_binary


def default_uir_affine_binary() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "UIR" / "build" / "uir_affine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m uir.cli.v2_affine_pipeline",
        description=(
            "Run the reproducible v2 affine experiment: prepare real-stack 500^3 "
            "working volumes, generate transformed cases with uir_affine, register "
            "with regSift3D, and write metrics/report."
        ),
    )
    parser.add_argument(
        "--input-stack",
        type=Path,
        default=Path("UIR/resources/real3d_pair_another/after_real_new_var"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("UIR/runs/v2/affine_real_big_500"),
    )
    parser.add_argument("--target-size", type=int, default=500)
    parser.add_argument("--binary", type=Path, default=None, help="regSift3D binary.")
    parser.add_argument("--uir-affine-binary", type=Path, default=None)
    parser.add_argument("--omp-num-threads", type=int, default=None)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Debug escape hatch. Omit for the full 10-case run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    binary = args.binary if args.binary is not None else default_regsift3d_binary()
    uir_affine = (
        args.uir_affine_binary
        if args.uir_affine_binary is not None
        else default_uir_affine_binary()
    )

    if args.omp_num_threads is None:
        raw_threads = os.environ.get("OMP_NUM_THREADS")
        omp_num_threads = int(raw_threads) if raw_threads else None
    else:
        omp_num_threads = args.omp_num_threads

    if not binary.exists():
        raise SystemExit(f"regSift3D binary not found: {binary}")
    if not uir_affine.exists():
        raise SystemExit(f"uir_affine binary not found: {uir_affine}")

    summary = run_affine_pipeline(
        input_stack=args.input_stack,
        out_dir=args.out_dir,
        binary=binary,
        uir_affine_binary=uir_affine,
        target_size=args.target_size,
        max_cases=args.max_cases,
        omp_num_threads=omp_num_threads,
    )
    print(json.dumps(
        {
            "out_dir": summary["out_dir"],
            "case_count": summary["case_count"],
            "report": str(Path(str(summary["out_dir"])) / "REPORT.md"),
            "summary": str(Path(str(summary["out_dir"])) / "summary.json"),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
