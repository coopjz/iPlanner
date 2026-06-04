#!/usr/bin/env bash
set -euo pipefail
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EXPERIMENT_DIRECTORY="$PKG_DIR/iplanner"
export IP_TRAINING_CONFIG="${IP_TRAINING_CONFIG:-$PKG_DIR/config/training_config_go2_odin.json}"
export WANDB_MODE="${WANDB_MODE:-offline}"
cd "$PKG_DIR/iplanner"
exec /home/unitree/miniforge3/envs/dog-nav-jetson/bin/python training_run.py "$@"
