#!/usr/bin/env bash
set -euo pipefail
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EXPERIMENT_DIRECTORY="$PKG_DIR/iplanner"
export IP_TRAINING_CONFIG="${IP_TRAINING_CONFIG:-$PKG_DIR/config/training_config_go2_odin_real_depth_raycast2d.json}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-ERROR}"
DEFAULT_IPLANNER_PYTHON="/home/cooper/miniforge3/envs/iplanner/bin/python"
if [[ -n "${IP_PYTHON:-}" ]]; then
  PYTHON_BIN="$IP_PYTHON"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "iplanner" && -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  PYTHON_BIN="$DEFAULT_IPLANNER_PYTHON"
fi
cd "$PKG_DIR/iplanner"
exec "$PYTHON_BIN" training_run.py "$@"
