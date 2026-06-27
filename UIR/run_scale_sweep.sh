#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-${workspace_dir}/.venv/bin/python}"
python_src_dir="${script_dir}/src"
runs_root="${UIR_RUNS_ROOT:-${script_dir}/runs}"
mpl_config_dir="${runs_root}/summary/.matplotlib"
variance="${SCALE_SWEEP_VARIANCE:-25}"

export PYTHONPATH="${python_src_dir}${PYTHONPATH:+:${PYTHONPATH}}"

default_scales=(1 1.25 1.5 2 3 4 6 8 10)

if [[ $# -gt 0 ]]; then
  scales=("$@")
else
  scales=("${default_scales[@]}")
fi

for scale_factor in "${scales[@]}"; do
  echo
  echo "=== Running isotropic scale ${scale_factor}, variance ${variance} ==="
  if ! "${script_dir}/run_scale_case.sh" "${scale_factor}" "${variance}"; then
    echo "Scale ${scale_factor} failed registration; keeping failure summary and continuing." >&2
  fi
done

mkdir -p "${mpl_config_dir}"
MPLCONFIGDIR="${mpl_config_dir}" "${python_bin}" -m uir.cli.summarize_synthetic_runs "${runs_root}"
