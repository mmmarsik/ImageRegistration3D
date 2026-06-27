#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [scale_factor] [variance]" >&2
  exit 1
fi

scale_factor="${1:-${SCALE_FACTOR:-2}}"
variance="${2:-${SCALE_SWEEP_VARIANCE:-25}}"

"${PYTHON_BIN:-${script_dir}/../.venv/bin/python}" - "$scale_factor" <<'PY'
import sys

try:
    scale = float(sys.argv[1])
except ValueError:
    print(f"Scale factor must be numeric, got: {sys.argv[1]}", file=sys.stderr)
    raise SystemExit(2)

if scale <= 0.0:
    print(f"Scale factor must be positive, got: {sys.argv[1]}", file=sys.stderr)
    raise SystemExit(2)
PY

export UIR_ROT_X_DEG="${UIR_ROT_X_DEG:-0}"
export UIR_ROT_Y_DEG="${UIR_ROT_Y_DEG:-0}"
export UIR_ROT_Z_DEG="${UIR_ROT_Z_DEG:-0}"
export UIR_SCALE_X="${scale_factor}"
export UIR_SCALE_Y="${scale_factor}"
export UIR_SCALE_Z="${scale_factor}"
export UIR_SH_XY="${UIR_SH_XY:-0}"
export UIR_SH_XZ="${UIR_SH_XZ:-0}"
export UIR_SH_YX="${UIR_SH_YX:-0}"
export UIR_SH_YZ="${UIR_SH_YZ:-0}"
export UIR_SH_ZX="${UIR_SH_ZX:-0}"
export UIR_SH_ZY="${UIR_SH_ZY:-0}"
export UIR_TX="${UIR_TX:-0}"
export UIR_TY="${UIR_TY:-0}"
export UIR_TZ="${UIR_TZ:-0}"

"${script_dir}/run_variance_case.sh" "${variance}"
