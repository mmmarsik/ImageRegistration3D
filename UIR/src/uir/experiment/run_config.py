"""Write a per-run ``run_config.json`` sidecar next to a run's outputs.

This is purely additive: it writes ONE new file, ``run_config.json``, into a run
directory, recording the scenario, the scalar parameters, and the resolved input
paths for that run. It never reads, rewrites, or deletes any existing artifact
(``summary.json`` and friends are untouched). Old runs that lack the file are
unaffected; aggregators may treat it as optional.

The file makes ``runs/`` self-describing: today a run's identity is reconstructed
from its folder name plus whatever flat fields the reporter happened to write.
``run_config.json`` records the exact scenario + parameters + the input stack/
volume paths that produced the run, so reproduction does not depend on parsing
the directory slug.

CLI
---
``python -m uir.experiment.run_config <run_dir> --scenario NAME``
``  [--param KEY=VALUE ...] [--input KEY=PATH ...]``

Values are stored as strings (the bash callers already hold them as strings).
Numeric-looking params are left as strings on purpose: this is a faithful record
of what the script passed, not a typed config object.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


RUN_CONFIG_FILENAME = "run_config.json"


def build_run_config(
    scenario: str,
    params: Mapping[str, str],
    inputs: Mapping[str, str],
) -> dict[str, object]:
    """Assemble the run-config payload (no I/O).

    Stable top-level shape: ``scenario`` (str), ``params`` (dict), ``inputs``
    (dict of resolved input paths). Kept intentionally small and flat.
    """

    return {
        "scenario": scenario,
        "params": dict(params),
        "inputs": dict(inputs),
    }


def write_run_config(
    run_dir: Path | str,
    scenario: str,
    params: Mapping[str, str],
    inputs: Mapping[str, str],
) -> Path:
    """Write ``<run_dir>/run_config.json`` and return its path.

    Creates ``run_dir`` if missing. Only ever writes the single new sidecar file;
    no existing file is read or modified.
    """

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / RUN_CONFIG_FILENAME
    payload = build_run_config(scenario, params, inputs)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return out_path


def _parse_kv(pairs: list[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in pairs:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"{flag} expects KEY=VALUE, got: {item!r}")
        out[key] = value
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m uir.experiment.run_config",
        description="Write a per-run run_config.json sidecar (additive; no existing artifact is touched).",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="A scalar run parameter (repeatable).",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="A resolved input path used by the run (repeatable).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    params = _parse_kv(args.param, "--param")
    inputs = _parse_kv(args.input, "--input")
    out_path = write_run_config(args.run_dir, args.scenario, params, inputs)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
