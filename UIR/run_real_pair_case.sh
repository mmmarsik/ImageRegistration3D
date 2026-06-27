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
sift_peak_thresh="${SIFT_PEAK_THRESH:-}"
sift_corner_thresh="${SIFT_CORNER_THRESH:-}"
sift_nn_thresh="${SIFT_NN_THRESH:-}"
sift_err_thresh="${SIFT_ERR_THRESH:-}"
sift_num_iter="${SIFT_NUM_ITER:-}"
model_consistent_threshold="${MODEL_CONSISTENT_THRESHOLD:-${sift_err_thresh:-5}}"
intensity_cuboid_radius="${INTENSITY_CUBOID_RADIUS:-5}"
png_stack_nifti_dtype="${PNG_STACK_NIFTI_DTYPE:-float32}"
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

ensure_sift3d_binary "${workspace_dir}" "${sift_build_dir}" "${nifti_install_dir}" "${build_jobs}"

export OMP_NUM_THREADS="${omp_num_threads}"
export_uir_pythonpath "${python_src_dir}"

read -r pair_tag case_tag < <(
  "${python_bin}" -m uir.cli.run_name real_pair \
    --before-stack-dir "${before_stack_dir}" \
    --pair-tag "${REAL_PAIR_TAG:-}"
)

png_stack_args=(
  "${voxel_spacing_x}"
  "${voxel_spacing_y}"
  "${voxel_spacing_z}"
)
before_png_stack_args=("${png_stack_args[@]}")
after_png_stack_args=("${png_stack_args[@]}")
summary_args=()
reg_sift_args=()

if [[ -n "${sift_peak_thresh}" ]]; then
  reg_sift_args+=(--peak_thresh "${sift_peak_thresh}")
fi
if [[ -n "${sift_corner_thresh}" ]]; then
  reg_sift_args+=(--corner_thresh "${sift_corner_thresh}")
fi
if [[ -n "${sift_nn_thresh}" ]]; then
  reg_sift_args+=(--nn_thresh "${sift_nn_thresh}")
fi
if [[ -n "${sift_err_thresh}" ]]; then
  reg_sift_args+=(--err_thresh "${sift_err_thresh}")
fi
if [[ -n "${sift_num_iter}" ]]; then
  reg_sift_args+=(--num_iter "${sift_num_iter}")
fi

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

case "${png_stack_nifti_dtype}" in
  float32|uint8|uint16)
    ;;
  *)
    echo "PNG_STACK_NIFTI_DTYPE must be one of: float32, uint8, uint16; got: ${png_stack_nifti_dtype}" >&2
    exit 1
    ;;
esac

run_dir="${runs_root}/real_pair/${pair_tag}/${case_tag}"
before_nii="${run_dir}/before.nii"
after_nii="${run_dir}/after.nii"
matches_csv="${run_dir}/matches.csv"
transform_csv="${run_dir}/transform.csv"
summary_json="${run_dir}/summary.json"

rm -rf "${run_dir}"
mkdir -p "${run_dir}"

"${python_bin}" -m uir.experiment.run_config "${run_dir}" \
  --scenario real_pair \
  --param "pair_tag=${pair_tag}" \
  --param "case_tag=${case_tag}" \
  --param "roi_size=${roi_size}" \
  --param "png_stack_nifti_dtype=${png_stack_nifti_dtype}" \
  --param "voxel_spacing_x=${voxel_spacing_x}" \
  --param "voxel_spacing_y=${voxel_spacing_y}" \
  --param "voxel_spacing_z=${voxel_spacing_z}" \
  --param "model_consistent_threshold=${model_consistent_threshold}" \
  --param "intensity_cuboid_radius=${intensity_cuboid_radius}" \
  --param "sift_peak_thresh=${sift_peak_thresh}" \
  --param "sift_corner_thresh=${sift_corner_thresh}" \
  --param "sift_nn_thresh=${sift_nn_thresh}" \
  --param "sift_err_thresh=${sift_err_thresh}" \
  --param "sift_num_iter=${sift_num_iter}" \
  --input "before_stack_dir=${before_stack_dir}" \
  --input "after_stack_dir=${after_stack_dir}" >/dev/null

"${python_bin}" -m uir.cli.png_stack_to_nifti \
  "${before_stack_dir}" \
  "${before_nii}" \
  "${before_png_stack_args[@]}" \
  --dtype "${png_stack_nifti_dtype}"

"${python_bin}" -m uir.cli.png_stack_to_nifti \
  "${after_stack_dir}" \
  "${after_nii}" \
  "${after_png_stack_args[@]}" \
  --dtype "${png_stack_nifti_dtype}"

reg_sift_cmd=(
  "${python_bin}" -m uir.registration
  --binary "${sift_build_dir}/bin/regSift3D"
  --matches "${matches_csv}"
  --transform "${transform_csv}"
  --reference "${before_nii}"
  --moving "${after_nii}"
)
if [[ ${#reg_sift_args[@]} -gt 0 ]]; then
  reg_sift_cmd+=(-- "${reg_sift_args[@]}")
fi

set +e
"${reg_sift_cmd[@]}"
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
  --model-consistent-threshold "${model_consistent_threshold}"
  --intensity-cuboid-radius "${intensity_cuboid_radius}"
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
echo "PNG stack NIfTI dtype: ${png_stack_nifti_dtype}"
if [[ ${#reg_sift_args[@]} -gt 0 ]]; then
  echo "SIFT3D args: ${reg_sift_args[*]}"
else
  echo "SIFT3D args: defaults"
fi
echo "Model-consistent threshold: ${model_consistent_threshold}"
echo "Intensity cuboid radius: ${intensity_cuboid_radius}"
echo "Run dir: ${run_dir}"
echo "Before stack: ${before_stack_dir}"
echo "After stack: ${after_stack_dir}"
echo "Before NIfTI: ${before_nii}"
echo "After NIfTI: ${after_nii}"
echo "Matches: ${matches_csv}"
echo "Transform: ${transform_csv}"
echo "Summary: ${summary_json}"
