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
source_stack_dir="${SOURCE_STACK_DIR:-${script_dir}/resources/bhi_2_2.32um_voi}"
ratio="${1:-${RESOLUTION_RATIO:-10}}"
crop_size="${CROP_SIZE:-250}"
blur_sigma_xy="${BLUR_SIGMA_XY:-1.0}"
awgn_variance="${AWGN_VARIANCE:-25}"
awgn_seed="${AWGN_SEED:-42}"
high_spacing="${HIGH_SPACING:-1.0}"
sift_err_thresh="${SIFT_ERR_THRESH:-}"
sift_num_iter="${SIFT_NUM_ITER:-}"
sift_nn_thresh="${SIFT_NN_THRESH:-}"

if ! [[ "${ratio}" =~ ^[0-9]+$ ]] || [[ "${ratio}" -le 0 ]]; then
  echo "Resolution ratio must be a positive integer, got: ${ratio}" >&2
  exit 1
fi
if ! [[ "${crop_size}" =~ ^[0-9]+$ ]] || [[ "${crop_size}" -le 0 ]]; then
  echo "CROP_SIZE must be a positive integer, got: ${crop_size}" >&2
  exit 1
fi

ensure_python_bin "${python_bin}"
ensure_stack_dir "${source_stack_dir}" "Source stack directory"
ensure_nifti_install "${nifti_install_dir}"
ensure_sift3d_binary "${workspace_dir}" "${sift_build_dir}" "${nifti_install_dir}" "${build_jobs}"

export OMP_NUM_THREADS="${omp_num_threads}"
export_uir_pythonpath "${python_src_dir}"

read -r blur_slug variance_padded < <(
  "${python_bin}" -m uir.cli.run_name resolution \
    --blur-sigma-xy "${blur_sigma_xy}" \
    --awgn-variance "${awgn_variance}"
)

case_tag="ratio${ratio}_crop${crop_size}_blur${blur_slug}_awgn_var${variance_padded}"
run_dir="${runs_root}/resolution_pair/${case_tag}"
metadata_json="${run_dir}/resolution_pair.json"
low_whole_nii="${run_dir}/low_res_whole.nii"
high_crop_clean_nii="${run_dir}/high_res_crop_clean.nii"
high_crop_blurred_nii="${run_dir}/high_res_crop_blur_sigma_${blur_slug}.nii"
high_crop_noisy_nii="${run_dir}/high_res_crop_blur_sigma_${blur_slug}_noise_var_${variance_padded}.nii"
expected_transform_csv="${run_dir}/expected_high_crop_to_low_whole_transform.csv"
mpl_config_dir="${run_dir}/.matplotlib"

rm -rf "${run_dir}"
mkdir -p "${run_dir}" "${mpl_config_dir}"

"${python_bin}" -m uir.experiment.run_config "${run_dir}" \
  --scenario resolution_pair \
  --param "ratio=${ratio}" \
  --param "crop_size=${crop_size}" \
  --param "blur_sigma_xy=${blur_sigma_xy}" \
  --param "blur_slug=${blur_slug}" \
  --param "awgn_variance=${awgn_variance}" \
  --param "variance_padded=${variance_padded}" \
  --param "awgn_seed=${awgn_seed}" \
  --param "high_spacing=${high_spacing}" \
  --input "source_stack_dir=${source_stack_dir}" >/dev/null

"${python_bin}" -m uir.cli.make_resolution_pair \
  "${source_stack_dir}" \
  "${run_dir}" \
  --ratio "${ratio}" \
  --crop-size "${crop_size}" "${crop_size}" "${crop_size}" \
  --high-spacing "${high_spacing}" >/dev/null

"${python_bin}" -m uir.cli.apply_gaussian_blur \
  "${high_crop_clean_nii}" \
  "${high_crop_blurred_nii}" \
  --sigma-xy "${blur_sigma_xy}"

"${python_bin}" -m uir.cli.add_gaussian_noise \
  "${high_crop_blurred_nii}" \
  "${high_crop_noisy_nii}" \
  --variance "${awgn_variance}" \
  --seed "${awgn_seed}"

reg_sift_args=()
if [[ -n "${sift_err_thresh}" ]]; then
  reg_sift_args+=(--err_thresh "${sift_err_thresh}")
fi
if [[ -n "${sift_num_iter}" ]]; then
  reg_sift_args+=(--num_iter "${sift_num_iter}")
fi
if [[ -n "${sift_nn_thresh}" ]]; then
  reg_sift_args+=(--nn_thresh "${sift_nn_thresh}")
fi

mode="resample"
mode_dir="${run_dir}/${mode}"
matches_csv="${mode_dir}/matches.csv"
transform_csv="${mode_dir}/transform.csv"
mkdir -p "${mode_dir}"

reg_cmd=(
  "${python_bin}" -m uir.registration
  --binary "${sift_build_dir}/bin/regSift3D"
  --resample
  --matches "${matches_csv}"
  --transform "${transform_csv}"
  --reference "${low_whole_nii}"
  --moving "${high_crop_noisy_nii}"
)
if [[ ${#reg_sift_args[@]} -gt 0 ]]; then
  reg_cmd+=(-- "${reg_sift_args[@]}")
fi

set +e
"${reg_cmd[@]}"
reg_exit_code=$?
set -e

MPLCONFIGDIR="${mpl_config_dir}" "${python_bin}" -m uir.cli.summarize_resolution_pair_case \
  "${run_dir}" \
  --mode "${mode}" \
  --matches-path "${matches_csv}" \
  --transform-path "${transform_csv}" \
  --expected-transform-path "${expected_transform_csv}" \
  --metadata-path "${metadata_json}" \
  --reg-exit-code "${reg_exit_code}"

echo
echo "Done."
echo "Resolution ratio: ${ratio}"
echo "Crop size: ${crop_size}"
echo "Blur sigma XY: ${blur_sigma_xy}"
echo "AWGN variance: ${awgn_variance}"
echo "Run dir: ${run_dir}"
echo "Low-res whole: ${low_whole_nii}"
echo "High-res crop noisy: ${high_crop_noisy_nii}"
echo "Expected transform: ${expected_transform_csv}"
