#!/usr/bin/env bash
set -euo pipefail
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EXPERIMENT_DIRECTORY="$PKG_DIR/iplanner"
export IP_DATA_GENERATION_CONFIG="${IP_DATA_GENERATION_CONFIG:-$PKG_DIR/config/data_generation_go2_odin.json}"
cd "$PKG_DIR/iplanner"
exec /home/unitree/miniforge3/envs/dog-nav-jetson/bin/python data_generation.py "$@"
