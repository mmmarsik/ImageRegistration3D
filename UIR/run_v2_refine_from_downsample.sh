#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH:-}:UIR/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"

.venv/bin/python -m uir.cli.v2_refine_from_downsample \
  --base-run-dir UIR/runs/v2/affine_real_big_500 \
  --out-dir UIR/runs/v2/refine_from_downsample_500 \
  --binary SIFT3D/build/bin/regSift3D \
  --omp-num-threads "$OMP_NUM_THREADS" \
  --cases \
    t02_translation_medium \
    t03_rotation_z_small \
    t05_rotation_xyz_medium \
    t06_scale_up_small \
    t09_rotation_translation \
    t10_combo_hard
