from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from uir.registration.base import RegistrationResult


@dataclass(frozen=True)
class RegSift3D:
    binary: Path
    resample: bool = False

    def build_command(
        self,
        reference_nifti: Path,
        moving_nifti: Path,
        *,
        matches_path: Path,
        transform_path: Path,
        extra_args: Sequence[str] = (),
    ) -> list[str]:

        cmd: list[str] = [str(self.binary)]
        if self.resample:
            cmd.append("--resample")
        cmd.extend(str(arg) for arg in extra_args)
        cmd.extend(
            [
                "--matches",
                str(matches_path),
                "--transform",
                str(transform_path),
                str(reference_nifti),
                str(moving_nifti),
            ]
        )
        return cmd

    def register(
        self,
        reference_nifti: Path,
        moving_nifti: Path,
        *,
        matches_path: Path,
        transform_path: Path,
        extra_args: Sequence[str] = (),
    ) -> RegistrationResult:
        cmd = self.build_command(
            reference_nifti,
            moving_nifti,
            matches_path=matches_path,
            transform_path=transform_path,
            extra_args=extra_args,
        )
        completed = subprocess.run(cmd, check=False)
        return RegistrationResult(
            matches_path=Path(matches_path),
            transform_path=Path(transform_path),
            exit_code=completed.returncode,
        )
