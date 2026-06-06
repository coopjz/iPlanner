#!/usr/bin/env bash
set -euo pipefail
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_IPLANNER_PYTHON="/home/cooper/miniforge3/envs/iplanner/bin/python"
if [[ -n "${IP_PYTHON:-}" ]]; then
  PYTHON_BIN="$IP_PYTHON"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "iplanner" && -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  PYTHON_BIN="$DEFAULT_IPLANNER_PYTHON"
fi

BAG_DIR="${BAG_DIR:-/media/cooper/XiangruT7/go2_odin_finetune}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PKG_DIR/iplanner/data/CollectedData}"
COLLECT_LIST="${COLLECT_LIST:-$PKG_DIR/iplanner/data/collect_list_go2_odin_real.txt}"
TRAINING_LIST="${TRAINING_LIST:-$PKG_DIR/iplanner/data/training_list_go2_odin_real.txt}"
REFERENCE_ENV="${REFERENCE_ENV:-$PKG_DIR/iplanner/data/CollectedData/go2_odin_002_part1}"
INTRINSIC="${INTRINSIC:-$REFERENCE_ENV/depth_intrinsic.txt}"
CAMERA_EXTRINSIC="${CAMERA_EXTRINSIC:-$REFERENCE_ENV/camera_extrinsic.txt}"
ENV_PREFIX="${ENV_PREFIX:-go2_odin_real}"

ARGS=(
  --bag-dir "$BAG_DIR"
  --output-root "$OUTPUT_ROOT"
  --collect-list "$COLLECT_LIST"
  --training-list "$TRAINING_LIST"
  --intrinsic "$INTRINSIC"
  --camera-extrinsic "$CAMERA_EXTRINSIC"
  --env-prefix "$ENV_PREFIX"
)

if [[ -n "${MAX_FRAMES:-}" ]]; then
  ARGS+=(--max-frames "$MAX_FRAMES")
fi
if [[ -n "${FRAME_STRIDE:-}" ]]; then
  ARGS+=(--frame-stride "$FRAME_STRIDE")
fi
if [[ "${OVERWRITE:-false}" == "true" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${FAIL_ON_BAD_BAG:-false}" == "true" ]]; then
  ARGS+=(--fail-on-bad-bag)
fi

exec "$PYTHON_BIN" "$PKG_DIR/scripts/extract_go2_odin_real_bags.py" "${ARGS[@]}" "$@"
