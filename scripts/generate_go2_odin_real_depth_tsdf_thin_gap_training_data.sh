#!/usr/bin/env bash
set -euo pipefail
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EXPERIMENT_DIRECTORY="$PKG_DIR/iplanner"
export IP_DATA_GENERATION_CONFIG="${IP_DATA_GENERATION_CONFIG:-$PKG_DIR/config/data_generation_go2_odin_real_depth_tsdf_thin_gap.json}"
DEFAULT_IPLANNER_PYTHON="/home/cooper/miniforge3/envs/iplanner/bin/python"
if [[ -n "${IP_PYTHON:-}" ]]; then
  PYTHON_BIN="$IP_PYTHON"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "iplanner" && -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  PYTHON_BIN="$DEFAULT_IPLANNER_PYTHON"
fi
cd "$PKG_DIR/iplanner"
exec "$PYTHON_BIN" data_generation.py "$@"
