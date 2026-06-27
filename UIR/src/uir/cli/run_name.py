"""Thin CLI that generates run-name tags/slugs for the case scripts.

This is purely a *naming* helper: it formats the directory-name slugs and tags
that the bash case scripts used to compute with inline Python heredocs. It does
NO registration and NO volume I/O. Centralizing the slug formatting here keeps
the run-folder names byte-for-byte identical to the old heredocs while removing
the duplicated, must-stay-in-sync logic from the scripts.

The formatting rules are copied verbatim from the previous inline snippets in
``UIR/run_variance_case.sh`` and ``UIR/run_resolution_pair_case.sh`` so existing
``UIR/runs/`` folder names (and the aggregators that parse them) are unaffected.

Subcommands
-----------
- ``variance``  -> prints ``<degradation> <case_tag> <blur_sigma_xy> <blur_slug>``
- ``resolution`` -> prints ``<blur_slug> <variance_padded>``
- ``real_pair`` -> prints ``<pair_tag> <case_tag>``

Each subcommand prints one space-separated line, matching what the scripts read
with ``read -r ...`` from the old heredoc output.
"""

from __future__ import annotations

import argparse
import os
import sys


def _blur_slug(sigma: float) -> str:
    """Blur-sigma slug, identical to the old heredocs.

    ``f"{sigma:.2f}".rstrip("0").rstrip(".").replace(".", "p")`` — e.g. 1.0 -> "1",
    1.5 -> "1p5", 0.75 -> "0p75".
    """

    return f"{sigma:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def _emit(tokens: list[str]) -> None:
    # Match the heredocs' ``print(a, b, ...)`` exactly: space-separated, newline.
    print(" ".join(tokens))


def _cmd_variance(args: argparse.Namespace) -> int:
    # Mirrors run_variance_case.sh heredoc. roi_size and variance_padded are
    # passed through verbatim (variance is already %04d-padded by the script).
    roi_size = args.roi_size
    variance_padded = args.variance_padded
    raw_sigma = args.blur_sigma_xy

    try:
        sigma = float(raw_sigma)
    except ValueError:
        print(f"Invalid BLUR_SIGMA_XY: {raw_sigma}", file=sys.stderr)
        return 2

    if sigma < 0.0:
        print(f"BLUR_SIGMA_XY must be non-negative, got: {raw_sigma}", file=sys.stderr)
        return 2

    if sigma == 0.0:
        _emit(["awgn", f"roi{roi_size}_awgn_var{variance_padded}", "0", "0"])
    else:
        slug = _blur_slug(sigma)
        _emit(
            [
                "blur_awgn",
                f"roi{roi_size}_blur{slug}_awgn_var{variance_padded}",
                f"{sigma:.12g}",
                slug,
            ]
        )
    return 0


def _cmd_resolution(args: argparse.Namespace) -> int:
    # Mirrors run_resolution_pair_case.sh heredoc.
    sigma = float(args.blur_sigma_xy)
    variance = float(args.awgn_variance)
    blur = _blur_slug(sigma)
    variance_slug = (
        f"{int(round(variance)):04d}"
        if variance.is_integer()
        else str(variance).replace(".", "p")
    )
    _emit([blur, variance_slug])
    return 0


def _cmd_real_pair(args: argparse.Namespace) -> int:
    # Mirrors run_real_pair_case.sh's pure-bash naming:
    #   pair_tag = REAL_PAIR_TAG or "<basename(dirname(before))>_before_to_after"
    #   case_tag = "roi<N>" if roi_size set else "full_volume"
    if args.pair_tag:
        pair_tag = args.pair_tag
    else:
        parent = os.path.basename(os.path.dirname(args.before_stack_dir))
        pair_tag = f"{parent}_before_to_after"

    if args.roi_size:
        case_tag = f"roi{args.roi_size}"
    else:
        case_tag = "full_volume"

    _emit([pair_tag, case_tag])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m uir.cli.run_name",
        description="Generate run-name tags/slugs for case scripts (naming only).",
    )
    sub = parser.add_subparsers(dest="scenario", required=True)

    p_var = sub.add_parser("variance", help="Synthetic AWGN / blur+AWGN case naming.")
    p_var.add_argument("--roi-size", dest="roi_size", required=True)
    p_var.add_argument("--variance-padded", dest="variance_padded", required=True)
    p_var.add_argument("--blur-sigma-xy", dest="blur_sigma_xy", required=True)
    p_var.set_defaults(func=_cmd_variance)

    p_res = sub.add_parser("resolution", help="Resolution-pair blur/variance slugs.")
    p_res.add_argument("--blur-sigma-xy", dest="blur_sigma_xy", required=True)
    p_res.add_argument("--awgn-variance", dest="awgn_variance", required=True)
    p_res.set_defaults(func=_cmd_resolution)

    p_real = sub.add_parser("real_pair", help="Real before/after pair tags.")
    p_real.add_argument("--before-stack-dir", dest="before_stack_dir", default="")
    p_real.add_argument("--pair-tag", dest="pair_tag", default="")
    p_real.add_argument("--roi-size", dest="roi_size", default="")
    p_real.set_defaults(func=_cmd_real_pair)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
