#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-${workspace_dir}/.venv/bin/python}"
python_src_dir="${script_dir}/src"
runs_root="${UIR_RUNS_ROOT:-${script_dir}/runs}"
mpl_config_dir="${runs_root}/resolution_pair/summary/.matplotlib"

export PYTHONPATH="${python_src_dir}${PYTHONPATH:+:${PYTHONPATH}}"

default_ratios=(2 4 5 10 20 25)

if [[ $# -gt 0 ]]; then
  ratios=("$@")
else
  ratios=("${default_ratios[@]}")
fi

for ratio in "${ratios[@]}"; do
  echo
  echo "=== Running physical-resolution ratio ${ratio} ==="
  "${script_dir}/run_resolution_pair_case.sh" "${ratio}"
done

mkdir -p "${mpl_config_dir}"
MPLCONFIGDIR="${mpl_config_dir}" "${python_bin}" -m uir.cli.summarize_resolution_pair_runs "${runs_root}"
