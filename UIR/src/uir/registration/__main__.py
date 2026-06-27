"""CLI entrypoint for the regSift3D wrapper.

Invoked by the bash case scripts as::

    python -m uir.registration \
        --binary <regSift3D> [--resample] \
        --matches <matches.csv> --transform <transform.csv> \
        --reference <ref.nii> --moving <moving.nii> \
        [-- <verbatim regSift3D option flags...>]

Everything after a literal ``--`` separator is forwarded to ``regSift3D``
verbatim and in order (the conditional ``--peak_thresh`` / ``--corner_thresh`` /
``--nn_thresh`` / ``--err_thresh`` / ``--num_iter`` pairs). The wrapper exits
with the same exit code ``regSift3D`` returns, so the scripts keep their
existing ``reg_exit_code`` handling unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from uir.registration.regsift3d import RegSift3D


def _split_extra_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split argv at the first literal ``--`` into (parsed, verbatim)."""

    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parsed_argv, extra_args = _split_extra_args(raw)

    parser = argparse.ArgumentParser(
        prog="python -m uir.registration",
        description="Thin wrapper around the external regSift3D binary.",
    )
    parser.add_argument("--binary", type=Path, required=True, help="Path to the regSift3D executable.")
    parser.add_argument("--resample", action="store_true", help="Pass --resample as the first regSift3D flag.")
    parser.add_argument("--matches", type=Path, required=True, help="Output matches CSV path.")
    parser.add_argument("--transform", type=Path, required=True, help="Output transform CSV path.")
    parser.add_argument("--reference", type=Path, required=True, help="Reference NIfTI volume.")
    parser.add_argument("--moving", type=Path, required=True, help="Moving NIfTI volume.")
    args = parser.parse_args(parsed_argv)

    backend = RegSift3D(binary=args.binary, resample=args.resample)
    result = backend.register(
        args.reference,
        args.moving,
        matches_path=args.matches,
        transform_path=args.transform,
        extra_args=extra_args,
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
