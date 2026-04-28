#!/usr/bin/env bash

reset_cmake_build_dir_if_needed() {
  local build_dir="$1"
  local expected_source_dir="$2"
  local cache_file="${build_dir}/CMakeCache.txt"

  if [[ -f "${cache_file}" ]] && ! grep -Fq "CMAKE_HOME_DIRECTORY:INTERNAL=${expected_source_dir}" "${cache_file}"; then
    echo "Resetting stale CMake cache in ${build_dir}"
    rm -rf "${build_dir}"
  fi
}

reset_sift3d_build_dir_if_openmp_cache_is_broken() {
  local build_dir="$1"
  local cache_file="${build_dir}/CMakeCache.txt"

  if [[ ! -f "${cache_file}" ]]; then
    return 0
  fi

  if grep -Eq '^WITH_OpenMP:.*=ON$' "${cache_file}" && {
    grep -Eq '^OpenMP_TRY_COMPILE_RESULT:INTERNAL=FALSE$' "${cache_file}" ||
      grep -Eq '^OpenMP_(C|CXX)_(FLAGS|LIBRARY):STRING=$' "${cache_file}";
  }; then
    echo "Resetting broken OpenMP CMake cache in ${build_dir}"
    rm -rf "${build_dir}"
  fi
}

ensure_python_bin() {
  local python_bin="$1"
  if [[ ! -x "${python_bin}" ]]; then
    echo "Python not found: ${python_bin}" >&2
    exit 1
  fi
}

ensure_stack_dir() {
  local stack_dir="$1"
  local label="${2:-Stack directory}"
  if [[ ! -d "${stack_dir}" ]]; then
    echo "${label} not found: ${stack_dir}" >&2
    exit 1
  fi
}

ensure_nifti_install() {
  local nifti_install_dir="$1"
  if [[ ! -f "${nifti_install_dir}/share/cmake/NIFTI/NIFTIConfig.cmake" ]]; then
    echo "nifti_clib install not found: ${nifti_install_dir}" >&2
    exit 1
  fi
}

export_uir_pythonpath() {
  local python_src_dir="$1"
  export PYTHONPATH="${python_src_dir}${PYTHONPATH:+:${PYTHONPATH}}"
}

detect_libomp_prefix() {
  local candidate

  for candidate in /opt/homebrew/opt/libomp /usr/local/opt/libomp; do
    if [[ -f "${candidate}/include/omp.h" ]] && [[ -f "${candidate}/lib/libomp.dylib" || -f "${candidate}/lib/libomp.a" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

compiler_is_apple_clang() {
  local compiler_bin="$1"
  "${compiler_bin}" --version 2>/dev/null | grep -qi '^Apple clang version'
}

configure_sift3d() {
  local workspace_dir="$1"
  local sift_build_dir="$2"
  local nifti_install_dir="$3"
  local with_openmp="$4"
  local libomp_prefix=""
  local c_flags=""
  local cxx_flags=""
  local exe_linker_flags=""
  local shared_linker_flags=""
  local -a cmake_args=(
    -S "${workspace_dir}/SIFT3D"
    -B "${sift_build_dir}"
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_Matlab=OFF
    -DBUILD_EXAMPLES=OFF
    -DNIFTI_DIR="${nifti_install_dir}"
    -DWITH_OpenMP="${with_openmp}"
  )

  if [[ "${with_openmp}" == "ON" ]] && [[ "$(uname -s)" == "Darwin" ]]; then
    if compiler_is_apple_clang "${CC:-cc}" && compiler_is_apple_clang "${CXX:-c++}"; then
      libomp_prefix="$(detect_libomp_prefix || true)"
      if [[ -n "${libomp_prefix}" ]]; then
        c_flags="${CFLAGS:-}"
        cxx_flags="${CXXFLAGS:-}"
        exe_linker_flags="${LDFLAGS:-}"
        shared_linker_flags="${LDFLAGS:-}"

        [[ -n "${c_flags}" ]] && c_flags+=" "
        [[ -n "${cxx_flags}" ]] && cxx_flags+=" "
        [[ -n "${exe_linker_flags}" ]] && exe_linker_flags+=" "
        [[ -n "${shared_linker_flags}" ]] && shared_linker_flags+=" "

        c_flags+="-Xpreprocessor -fopenmp -I${libomp_prefix}/include"
        cxx_flags+="-Xpreprocessor -fopenmp -I${libomp_prefix}/include"
        exe_linker_flags+="-L${libomp_prefix}/lib -lomp"
        shared_linker_flags+="-L${libomp_prefix}/lib -lomp"

        cmake_args+=(
          "-DCMAKE_C_FLAGS=${c_flags}"
          "-DCMAKE_CXX_FLAGS=${cxx_flags}"
          "-DCMAKE_EXE_LINKER_FLAGS=${exe_linker_flags}"
          "-DCMAKE_SHARED_LINKER_FLAGS=${shared_linker_flags}"
        )
      fi
    fi
  fi

  cmake "${cmake_args[@]}"
}

build_sift3d() {
  local workspace_dir="$1"
  local sift_build_dir="$2"
  local nifti_install_dir="$3"
  local build_jobs="$4"

  reset_cmake_build_dir_if_needed "${sift_build_dir}" "${workspace_dir}/SIFT3D"
  reset_sift3d_build_dir_if_openmp_cache_is_broken "${sift_build_dir}"

  if ! configure_sift3d "${workspace_dir}" "${sift_build_dir}" "${nifti_install_dir}" ON; then
    echo "OpenMP-enabled SIFT3D build is unavailable; retrying without OpenMP."
    rm -rf "${sift_build_dir}"
    configure_sift3d "${workspace_dir}" "${sift_build_dir}" "${nifti_install_dir}" OFF
  fi

  cmake --build "${sift_build_dir}" -j "${build_jobs}"
}

build_uir() {
  local script_dir="$1"
  local uir_build_dir="$2"
  local build_jobs="$3"

  reset_cmake_build_dir_if_needed "${uir_build_dir}" "${script_dir}"
  cmake -S "${script_dir}" -B "${uir_build_dir}" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${uir_build_dir}" -j "${build_jobs}"
}
