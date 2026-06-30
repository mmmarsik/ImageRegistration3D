#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
cd "${repo_dir}"

python_bin="${PYTHON_BIN:-.venv/bin/python}"
input_stack="${INPUT_STACK:-UIR/resources/real3d_pair_another/after_real_new_var}"
out_dir="${OUT_DIR:-UIR/runs/v2/affine_real_big_500}"
target_size="${TARGET_SIZE:-500}"
omp_num_threads="${OMP_NUM_THREADS:-12}"
regsift3d_bin="${REGSIFT3D_BIN:-SIFT3D/build/bin/regSift3D}"
uir_affine_bin="${UIR_AFFINE_BIN:-UIR/build/uir_affine}"

export PYTHONPATH="${repo_dir}/UIR/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${omp_num_threads}"

cmd=(
  "${python_bin}" -m uir.cli.v2_affine_pipeline
  --input-stack "${input_stack}"
  --out-dir "${out_dir}"
  --target-size "${target_size}"
  --binary "${regsift3d_bin}"
  --uir-affine-binary "${uir_affine_bin}"
  --omp-num-threads "${omp_num_threads}"
)

if [[ -n "${MAX_CASES:-}" ]]; then
  cmd+=(--max-cases "${MAX_CASES}")
fi

echo "Input stack: ${input_stack}"
echo "Output dir:  ${out_dir}"
echo "Target size: ${target_size}^3"
echo "OMP threads: ${omp_num_threads}"
echo "regSift3D:   ${regsift3d_bin}"
echo "uir_affine:  ${uir_affine_bin}"
echo
printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n\n'

"${cmd[@]}"

echo
echo "Report:  ${out_dir}/REPORT.md"
echo "Summary: ${out_dir}/summary.json"
