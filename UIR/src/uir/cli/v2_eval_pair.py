from __future__ import annotations

import argparse
import json
from pathlib import Path

from uir.v2.metrics import MODEL_CONSISTENT_THRESHOLD_DEFAULT, evaluate_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m uir.cli.v2_eval_pair",
        description="Compute match-residual metrics for an existing v2 run directory.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=MODEL_CONSISTENT_THRESHOLD_DEFAULT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = evaluate_run(args.run_dir, threshold=args.threshold)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
