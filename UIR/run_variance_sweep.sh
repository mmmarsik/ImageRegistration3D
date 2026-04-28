#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-${workspace_dir}/.venv/bin/python}"
python_src_dir="${script_dir}/src"
runs_root="${UIR_RUNS_ROOT:-${script_dir}/runs}"
mpl_config_dir="${runs_root}/summary/.matplotlib"

export PYTHONPATH="${python_src_dir}${PYTHONPATH:+:${PYTHONPATH}}"

default_variances=(5 10 15 20 25 30 35 40 45 50 100 300)

if [[ $# -gt 0 ]]; then
  variances=("$@")
else
  variances=("${default_variances[@]}")
fi

for variance in "${variances[@]}"; do
  echo
  echo "=== Running variance ${variance} ==="
  "${script_dir}/run_variance_case.sh" "${variance}"
done

mkdir -p "${mpl_config_dir}"
MPLCONFIGDIR="${mpl_config_dir}" "${python_bin}" -m uir.cli.summarize_synthetic_runs "${runs_root}"
