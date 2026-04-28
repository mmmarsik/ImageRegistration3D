#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "${script_dir}/.." && pwd)"

# shellcheck source=UIR/common.sh
source "${script_dir}/common.sh"

python_bin="${PYTHON_BIN:-${workspace_dir}/.venv/bin/python}"
python_src_dir="${script_dir}/src"
uir_build_dir="${UIR_BUILD_DIR:-${script_dir}/build}"
sift_build_dir="${SIFT_BUILD_DIR:-${workspace_dir}/SIFT3D/build}"
nifti_install_dir="${NIFTI_INSTALL_DIR:-${workspace_dir}/nifti_clib/install}"
runs_root="${UIR_RUNS_ROOT:-${script_dir}/runs}"
build_jobs="${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
omp_num_threads="${OMP_NUM_THREADS:-${build_jobs}}"
roi_size="${ROI_SIZE:-250}"
blur_sigma_xy_raw="${BLUR_SIGMA_XY:-0}"
awgn_seed="${AWGN_SEED:-42}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <variance>" >&2
  exit 1
fi

variance_raw="$1"
if ! [[ "${variance_raw}" =~ ^[0-9]+$ ]]; then
  echo "Variance must be a non-negative integer, got: ${variance_raw}" >&2
  exit 1
fi

variance="${variance_raw}"
variance_padded="$(printf '%04d' "${variance}")"

source_stack_dir="${SOURCE_STACK_DIR:-${script_dir}/resources/bhi_2_2.32um_voi}"
ensure_python_bin "${python_bin}"
ensure_stack_dir "${source_stack_dir}" "Source stack directory"
ensure_nifti_install "${nifti_install_dir}"

read -r degradation case_tag blur_sigma_xy blur_slug < <(
  ROI_SIZE="${roi_size}" VARIANCE_PADDED="${variance_padded}" BLUR_SIGMA_XY="${blur_sigma_xy_raw}" "${python_bin}" -c '
import os
import sys

roi_size = os.environ["ROI_SIZE"]
variance_padded = os.environ["VARIANCE_PADDED"]
raw_sigma = os.environ["BLUR_SIGMA_XY"]

try:
    sigma = float(raw_sigma)
except ValueError:
    print(f"Invalid BLUR_SIGMA_XY: {raw_sigma}", file=sys.stderr)
    raise SystemExit(2)

if sigma < 0.0:
    print(f"BLUR_SIGMA_XY must be non-negative, got: {raw_sigma}", file=sys.stderr)
    raise SystemExit(2)

if sigma == 0.0:
    print("awgn", f"roi{roi_size}_awgn_var{variance_padded}", "0", "0")
else:
    slug = f"{sigma:.2f}".rstrip("0").rstrip(".").replace(".", "p")
    print("blur_awgn", f"roi{roi_size}_blur{slug}_awgn_var{variance_padded}", f"{sigma:.12g}", slug)
'
)

build_uir "${script_dir}" "${uir_build_dir}" "${build_jobs}"
build_sift3d "${workspace_dir}" "${sift_build_dir}" "${nifti_install_dir}" "${build_jobs}"

export OMP_NUM_THREADS="${omp_num_threads}"
export_uir_pythonpath "${python_src_dir}"

transform_tag="$("${uir_build_dir}/uir_affine" --transform-tag)"
transform_root="${runs_root}/${transform_tag}"
run_dir="${transform_root}/${case_tag}"
stack_dir="${run_dir}/transformed_png_stack"
volume_a_nii="${run_dir}/volume_A_roi${roi_size}.nii"
volume_b_clean_nii="${run_dir}/volume_B_roi${roi_size}_clean.nii"
volume_b_blurred_nii="${run_dir}/volume_B_roi${roi_size}_blur_sigma_${blur_slug}.nii"
if [[ "${degradation}" == "blur_awgn" ]]; then
  volume_b_noisy_nii="${run_dir}/volume_B_roi${roi_size}_blur_sigma_${blur_slug}_noise_var_${variance_padded}.nii"
else
  volume_b_noisy_nii="${run_dir}/volume_B_roi${roi_size}_noise_var_${variance_padded}.nii"
fi
volume_a_observation_json="${volume_a_nii}.observation.json"
matches_csv="${run_dir}/matches.csv"
transform_csv="${run_dir}/transform.csv"
mpl_config_dir="${run_dir}/.matplotlib"

mkdir -p "${transform_root}"
"${uir_build_dir}/uir_affine" --transform-description > "${transform_root}/transform_info.txt"

rm -rf "${run_dir}"
mkdir -p "${run_dir}" "${mpl_config_dir}"

"${uir_build_dir}/uir_affine" "${stack_dir}" "${run_dir}"

"${python_bin}" -m uir.cli.png_stack_to_nifti \
  "${source_stack_dir}" \
  "${volume_a_nii}" \
  1 1 1 \
  --roi-size "${roi_size}" "${roi_size}" "${roi_size}"

"${python_bin}" -m uir.cli.png_stack_to_nifti \
  "${stack_dir}" \
  "${volume_b_clean_nii}" \
  1 1 1 \
  --observation-model-like "${volume_a_nii}" \
  --roi-size "${roi_size}" "${roi_size}" "${roi_size}"

if [[ ! -f "${volume_a_observation_json}" ]]; then
  echo "Missing observation model for ${volume_a_nii}" >&2
  exit 1
fi

noise_reference_nii="${volume_b_clean_nii}"
if [[ "${degradation}" == "blur_awgn" ]]; then
  "${python_bin}" -m uir.cli.apply_gaussian_blur \
    "${volume_b_clean_nii}" \
    "${volume_b_blurred_nii}" \
    --sigma-xy "${blur_sigma_xy}"
  noise_reference_nii="${volume_b_blurred_nii}"
fi

"${python_bin}" -m uir.cli.add_gaussian_noise \
  "${noise_reference_nii}" \
  "${volume_b_noisy_nii}" \
  --variance "${variance}" \
  --seed "${awgn_seed}"

"${sift_build_dir}/bin/regSift3D" \
  --matches "${matches_csv}" \
  --transform "${transform_csv}" \
  "${volume_a_nii}" \
  "${volume_b_noisy_nii}"

MPLCONFIGDIR="${mpl_config_dir}" "${python_bin}" -m uir.cli.plot_single_case_report \
  "${run_dir}" \
  --roi-size "${roi_size}" "${roi_size}" "${roi_size}" \
  --source-stack-dir "${source_stack_dir}" \
  --noisy-path "${volume_b_noisy_nii}" \
  --matches-path "${matches_csv}" \
  --noise-reference-path "${noise_reference_nii}" \
  --degradation "${degradation}" \
  --transform-tag "${transform_tag}" \
  --blur-sigma-xy "${blur_sigma_xy}" \
  --awgn-variance "${variance}" \
  --awgn-seed "${awgn_seed}"

echo
echo "Done."
echo "Transform tag: ${transform_tag}"
echo "Transform root: ${transform_root}"
echo "Variance: ${variance}"
echo "Degradation: ${degradation}"
echo "Blur sigma XY: ${blur_sigma_xy}"
echo "AWGN seed: ${awgn_seed}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "Run dir: ${run_dir}"
echo "Reference NIfTI: ${volume_a_nii}"
echo "Clean transformed NIfTI: ${volume_b_clean_nii}"
if [[ "${degradation}" == "blur_awgn" ]]; then
  echo "Blurred transformed NIfTI: ${volume_b_blurred_nii}"
fi
echo "Noisy transformed NIfTI: ${volume_b_noisy_nii}"
echo "Matches: ${matches_csv}"
echo "Transform: ${transform_csv}"
echo "Plots: ${run_dir}/plots"
