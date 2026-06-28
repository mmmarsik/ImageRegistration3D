
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class RegistrationResult:
    matches_path: Path
    transform_path: Path
    exit_code: int


class Registrar(Protocol):
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
