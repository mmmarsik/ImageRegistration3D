"""Registration contract: one result struct and one Protocol.

Mirrors ``docs/audit/03_api_contracts.md`` ("Registration contract"). The
pipeline only consumes two output CSVs plus the subprocess exit code; the
registration step itself is an external subprocess, not a Python-pluggable
component. This file names that shape without changing any behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class RegistrationResult:
    """Outputs of a single registration invocation.

    Attributes mirror what the bash drivers already thread around as
    ``matches_csv`` / ``transform_csv`` / ``reg_exit_code``.
    """

    matches_path: Path
    transform_path: Path
    exit_code: int


class Registrar(Protocol):
    """A registration backend.

    Today implemented only by :class:`uir.registration.regsift3d.RegSift3D`, a
    thin subprocess wrapper around the external ``regSift3D`` binary.

    ``extra_args`` is forwarded to the backend verbatim and in order. Callers
    are responsible for assembling backend-specific option flags (e.g.
    ``--peak_thresh 0.1``) exactly as the backend expects them; the wrapper does
    not interpret them.
    """

    def register(
        self,
        reference_nifti: Path,
        moving_nifti: Path,
        *,
        matches_path: Path,
        transform_path: Path,
        extra_args: Sequence[str] = (),
    ) -> RegistrationResult:
        ...
