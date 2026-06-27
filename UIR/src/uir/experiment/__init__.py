"""Run-level result contracts for UIR experiments.

This package documents the on-disk `summary.json` wire format with types and
provides a lenient loader. It is purely additive: writers and readers in the
rest of the pipeline are NOT switched to it. See ``summary.py`` for details.
"""

from uir.experiment.run_config import (
    RUN_CONFIG_FILENAME,
    build_run_config,
    write_run_config,
)
from uir.experiment.summary import (
    MatchResidualStats,
    RunPaths,
    RunSummary,
    load_run_summary,
)

__all__ = [
    "MatchResidualStats",
    "RunPaths",
    "RunSummary",
    "load_run_summary",
    "RUN_CONFIG_FILENAME",
    "build_run_config",
    "write_run_config",
]
