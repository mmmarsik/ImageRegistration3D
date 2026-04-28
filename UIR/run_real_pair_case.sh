#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "${script_dir}/.." && pwd)"

# shellcheck source=UIR/common.sh
source "${script_dir}/common.sh"

python_bin="${PYTHON_BIN:-${workspace_dir}/.venv/bin/python}"
python_src_dir="${script_dir}/src"
sift_build_dir="${SIFT_BUILD_DIR:-${workspace_dir}/SIFT3D/build}"
nifti_install_dir="${NIFTI_INSTALL_DIR:-${workspace_dir}/nifti_clib/install}"
runs_root="${UIR_RUNS_ROOT:-${script_dir}/runs}"
build_jobs="${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
omp_num_threads="${OMP_NUM_THREADS:-${build_jobs}}"
roi_size="${ROI_SIZE:-}"
voxel_spacing_x="${VOXEL_SPACING_X:-1}"
voxel_spacing_y="${VOXEL_SPACING_Y:-1}"
voxel_spacing_z="${VOXEL_SPACING_Z:-1}"
before_stack_dir="${BEFORE_STACK_DIR:-${script_dir}/resources/real3d_pair/before_png_stack}"
after_stack_dir="${AFTER_STACK_DIR:-${script_dir}/resources/real3d_pair/after_png_stack}"

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [before_stack_dir after_stack_dir]" >&2
  exit 1
fi

if [[ $# -ge 1 ]]; then
  before_stack_dir="$1"
fi

if [[ $# -ge 2 ]]; then
  after_stack_dir="$2"
fi

ensure_python_bin "${python_bin}"
ensure_stack_dir "${before_stack_dir}" "Before stack directory"
ensure_stack_dir "${after_stack_dir}" "After stack directory"
ensure_nifti_install "${nifti_install_dir}"

build_sift3d "${workspace_dir}" "${sift_build_dir}" "${nifti_install_dir}" "${build_jobs}"

export OMP_NUM_THREADS="${omp_num_threads}"
export_uir_pythonpath "${python_src_dir}"

pair_tag="${REAL_PAIR_TAG:-$(basename "$(dirname "${before_stack_dir}")")_before_to_after}"
case_tag="full_volume"

png_stack_args=(
  "${voxel_spacing_x}"
  "${voxel_spacing_y}"
  "${voxel_spacing_z}"
)
before_png_stack_args=("${png_stack_args[@]}")
after_png_stack_args=("${png_stack_args[@]}")
summary_args=()

if [[ -n "${roi_size}" ]]; then
  if ! [[ "${roi_size}" =~ ^[0-9]+$ ]]; then
    echo "ROI_SIZE must be a positive integer, got: ${roi_size}" >&2
    exit 1
  fi
  read -r roi_x roi_y roi_z before_x before_y before_z after_x after_y after_z < <(
    "${python_bin}" -m uir.cli.plan_real_pair_crop \
      "${before_stack_dir}" \
      "${after_stack_dir}" \
      --roi-size "${roi_size}" "${roi_size}" "${roi_size}"
  )

  case_tag="roi${roi_size}"
  before_png_stack_args+=(--roi-size "${roi_x}" "${roi_y}" "${roi_z}" --roi-start "${before_x}" "${before_y}" "${before_z}")
  after_png_stack_args+=(--roi-size "${roi_x}" "${roi_y}" "${roi_z}" --roi-start "${after_x}" "${after_y}" "${after_z}")
  summary_args+=(
    --roi-size "${roi_x}" "${roi_y}" "${roi_z}"
    --before-roi-start "${before_x}" "${before_y}" "${before_z}"
    --after-roi-start "${after_x}" "${after_y}" "${after_z}"
  )
fi

run_dir="${runs_root}/real_pair/${pair_tag}/${case_tag}"
before_nii="${run_dir}/before.nii"
after_nii="${run_dir}/after.nii"
matches_csv="${run_dir}/matches.csv"
transform_csv="${run_dir}/transform.csv"
summary_json="${run_dir}/summary.json"

rm -rf "${run_dir}"
mkdir -p "${run_dir}"

"${python_bin}" -m uir.cli.png_stack_to_nifti \
  "${before_stack_dir}" \
  "${before_nii}" \
  "${before_png_stack_args[@]}"

"${python_bin}" -m uir.cli.png_stack_to_nifti \
  "${after_stack_dir}" \
  "${after_nii}" \
  "${after_png_stack_args[@]}"

set +e
"${sift_build_dir}/bin/regSift3D" \
  --matches "${matches_csv}" \
  --transform "${transform_csv}" \
  "${before_nii}" \
  "${after_nii}"
reg_exit_code=$?
set -e

summary_cmd=(
  "${python_bin}" -m uir.cli.summarize_real_pair_case
  "${run_dir}"
  --before-stack-dir "${before_stack_dir}"
  --after-stack-dir "${after_stack_dir}"
  --before-nifti "${before_nii}"
  --after-nifti "${after_nii}"
  --matches-path "${matches_csv}"
  --transform-path "${transform_csv}"
  --reg-exit-code "${reg_exit_code}"
)

if [[ ${#summary_args[@]} -gt 0 ]]; then
  summary_cmd+=("${summary_args[@]}")
fi

"${summary_cmd[@]}"

if [[ "${reg_exit_code}" -ne 0 ]]; then
  echo "regSift3D failed with exit code ${reg_exit_code}" >&2
  exit "${reg_exit_code}"
fi

echo
echo "Done."
echo "Pair tag: ${pair_tag}"
echo "Case tag: ${case_tag}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "Run dir: ${run_dir}"
echo "Before stack: ${before_stack_dir}"
echo "After stack: ${after_stack_dir}"
echo "Before NIfTI: ${before_nii}"
echo "After NIfTI: ${after_nii}"
echo "Matches: ${matches_csv}"
echo "Transform: ${transform_csv}"
echo "Summary: ${summary_json}"
