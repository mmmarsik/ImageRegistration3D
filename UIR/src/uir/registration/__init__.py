"""Registration backend wrappers for the UIR pipeline.

This package encapsulates the invocation of the external ``regSift3D`` binary so
that the bash case scripts and any future Python driver build the *same* command
line in one place. The flags forwarded to ``regSift3D`` are passed through
verbatim — this wrapper never adds, drops, reorders, or rewrites them.

The single genuine polymorphism point (swapping SIFT3D for another registration
tool later) is the :class:`Registrar` ``Protocol``. Today there is exactly one
implementation, :class:`RegSift3D`.
"""

from __future__ import annotations

from uir.registration.base import Registrar, RegistrationResult
from uir.registration.regsift3d import RegSift3D

__all__ = ["Registrar", "RegistrationResult", "RegSift3D"]
