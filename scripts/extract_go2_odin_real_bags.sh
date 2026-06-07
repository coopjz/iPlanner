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
if [[ "${DISABLE_SELF_FILTER:-false}" == "true" ]]; then
  ARGS+=(--disable-self-filter)
fi
for opt in \
  SELF_FILTER_X_MIN:self-filter-x-min \
  SELF_FILTER_X_MAX:self-filter-x-max \
  SELF_FILTER_Y_MIN:self-filter-y-min \
  SELF_FILTER_Y_MAX:self-filter-y-max \
  SELF_FILTER_Z_MIN:self-filter-z-min \
  SELF_FILTER_Z_MAX:self-filter-z-max; do
  env_name="${opt%%:*}"
  arg_name="${opt#*:}"
  value="${!env_name:-}"
  if [[ -n "$value" ]]; then
    ARGS+=(--"$arg_name" "$value")
  fi
done
if [[ "${OVERWRITE:-false}" == "true" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${FAIL_ON_BAD_BAG:-false}" == "true" ]]; then
  ARGS+=(--fail-on-bad-bag)
fi

exec "$PYTHON_BIN" "$PKG_DIR/scripts/extract_go2_odin_real_bags.py" "${ARGS[@]}" "$@"
