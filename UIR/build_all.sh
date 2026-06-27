#!/usr/bin/env bash
set -euo pipefail

# Build the native binaries (uir_affine and SIFT3D's regSift3D) once per
# session, decoupled from the per-case run scripts. The case scripts source the
# helpers below (ensure_uir_binary / ensure_sift3d_binary) and only build when a
# binary is missing, so a sweep no longer reconfigures+rebuilds on every point.
#
# Usage:
#   UIR/build_all.sh            # build both, skipping any binary already present
#   FORCE_REBUILD=1 UIR/build_all.sh   # rebuild both unconditionally
#   UIR/build_all.sh uir        # build only uir_affine
#   UIR/build_all.sh sift3d     # build only regSift3D
#
# Honors the same env overrides as the case scripts: PYTHON_BIN, UIR_BUILD_DIR,
# SIFT_BUILD_DIR, NIFTI_INSTALL_DIR, BUILD_JOBS, FORCE_REBUILD.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "${script_dir}/.." && pwd)"

# shellcheck source=UIR/common.sh
source "${script_dir}/common.sh"

uir_build_dir="${UIR_BUILD_DIR:-${script_dir}/build}"
sift_build_dir="${SIFT_BUILD_DIR:-${workspace_dir}/SIFT3D/build}"
nifti_install_dir="${NIFTI_INSTALL_DIR:-${workspace_dir}/nifti_clib/install}"
build_jobs="${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

target="${1:-all}"

case "${target}" in
  all|uir|sift3d)
    ;;
  *)
    echo "Usage: $0 [all|uir|sift3d]" >&2
    exit 1
    ;;
esac

if [[ "${target}" == "all" || "${target}" == "uir" ]]; then
  ensure_uir_binary "${script_dir}" "${uir_build_dir}" "${build_jobs}"
fi

if [[ "${target}" == "all" || "${target}" == "sift3d" ]]; then
  ensure_nifti_install "${nifti_install_dir}"
  ensure_sift3d_binary "${workspace_dir}" "${sift_build_dir}" "${nifti_install_dir}" "${build_jobs}"
fi

echo "build_all: done (${target})."
