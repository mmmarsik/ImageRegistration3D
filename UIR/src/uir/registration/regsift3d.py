"""Subprocess wrapper around the external ``regSift3D`` binary.

CRITICAL: the command line built here must be byte-for-byte identical to the
inline invocations previously hardcoded in the three case scripts:

* ``run_variance_case.sh``::

      regSift3D --matches M --transform T REF MOVING

* ``run_real_pair_case.sh``::

      regSift3D [opt flags...] --matches M --transform T REF MOVING

* ``run_resolution_pair_case.sh``::

      regSift3D --resample [opt flags...] --matches M --transform T REF MOVING

The canonical argument order is therefore:

    <binary> [--resample] [extra_args...] --matches M --transform T REF MOVING

``extra_args`` (the conditional ``--peak_thresh`` / ``--corner_thresh`` /
``--nn_thresh`` / ``--err_thresh`` / ``--num_iter`` pairs) are forwarded
verbatim and in the order the caller supplies them. This wrapper never adds,
drops, reorders, or rewrites SIFT flags or thresholds.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from uir.registration.base import RegistrationResult


@dataclass(frozen=True)
class RegSift3D:
    """``regSift3D`` registration backend.

    Parameters
    ----------
    binary:
        Path to the ``regSift3D`` executable (``${SIFT_BUILD_DIR}/bin/regSift3D``).
    resample:
        Pass ``--resample`` as the first flag (used by the resolution-pair
        scenario). Defaults to ``False`` to match the synthetic and real-pair
        scenarios.
    """

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
        """Assemble the exact argv passed to ``regSift3D``.

        Kept separate from :meth:`register` so the command can be inspected /
        diffed against the legacy inline command without running the binary.
        """

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
        # Mirror the bash drivers: do not raise on a non-zero exit; the exit code
        # is threaded into the per-run summary and the script decides what to do.
        completed = subprocess.run(cmd, check=False)
        return RegistrationResult(
            matches_path=Path(matches_path),
            transform_path=Path(transform_path),
            exit_code=completed.returncode,
        )
