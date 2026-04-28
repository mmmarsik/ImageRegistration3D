from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild synthetic per-run reports from existing transforms and volumes."
    )
    parser.add_argument("runs_root", type=Path, help="UIR/runs directory.")
    parser.add_argument(
        "--source-stack-dir",
        type=Path,
        default=Path("UIR/resources/bhi_2_2.32um_voi"),
        help="Original source PNG stack used by synthetic runs.",
    )
    parser.add_argument(
        "--only",
        type=Path,
        help="Refresh one run directory instead of all synthetic runs.",
    )
    return parser.parse_args()


def _summary_paths(args: argparse.Namespace) -> list[Path]:
    if args.only is not None:
        return [args.only / "plots" / "summary.json"]
    return sorted(args.runs_root.glob("*/**/plots/summary.json"))


def _str_path(summary: dict[str, object], key: str) -> str | None:
    value = summary.get(key)
    if value in (None, ""):
        return None
    return str(value)


def _local_path(value: str | Path) -> Path:
    path = Path(value)
    container_prefix = Path("/workspace/ImageRegistration3D")
    if path.is_absolute():
        try:
            relative = path.relative_to(container_prefix)
        except ValueError:
            return path
        return Path.cwd() / relative
    return path


def _command_for_summary(summary_path: Path, summary: dict[str, object], source_stack_dir: Path) -> list[str]:
    run_dir = _local_path(str(summary.get("run_dir") or summary_path.parent.parent))
    roi_size_xyz = [str(int(v)) for v in summary["roi_size_xyz"]]

    command = [
        sys.executable,
        "-m",
        "uir.cli.plot_single_case_report",
        str(run_dir),
        "--roi-size",
        *roi_size_xyz,
        "--source-stack-dir",
        str(source_stack_dir),
    ]

    optional_path_args = {
        "--noisy-path": "noisy_path",
        "--matches-path": "matches_path",
        "--noise-reference-path": "noise_reference_path",
    }
    for cli_arg, summary_key in optional_path_args.items():
        value = _str_path(summary, summary_key)
        if value is not None:
            command.extend([cli_arg, str(_local_path(value))])

    optional_scalar_args = {
        "--degradation": "degradation",
        "--transform-tag": "transform_tag",
        "--blur-sigma-xy": "blur_sigma_xy",
        "--awgn-variance": "awgn_variance",
        "--awgn-seed": "awgn_seed",
    }
    for cli_arg, summary_key in optional_scalar_args.items():
        value = summary.get(summary_key)
        if value not in (None, ""):
            command.extend([cli_arg, str(value)])

    return command


def main() -> int:
    args = parse_args()
    refreshed = 0

    for summary_path in _summary_paths(args):
        if not summary_path.exists():
            raise RuntimeError(f"Missing summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("run_kind") != "synthetic":
            continue
        command = _command_for_summary(summary_path, summary, args.source_stack_dir)
        subprocess.run(command, check=True)
        refreshed += 1

    print(f"Refreshed synthetic reports: {refreshed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
