#!/usr/bin/env bash
set -euo pipefail

# Record the minimum ROS1 bag needed to replay iPlanner Go2+Odin data collection.
# Assumes Odin and loamInterface/iPlanner TF publishers are already running.

BAG_DIR="${BAG_DIR:-$HOME/dog_nav_stack_demo/iPlanner/bags/go2_odin_finetune}"
BAG_PREFIX="${BAG_PREFIX:-go2_odin_finetune_$(date +%Y%m%d_%H%M%S)}"
SPLIT_SIZE_MB="${SPLIT_SIZE_MB:-4096}"
COMPRESS="${COMPRESS:-bz2}"
INCLUDE_RAW_ODIN="${INCLUDE_RAW_ODIN:-false}"
INCLUDE_GO2="${INCLUDE_GO2:-false}"

mkdir -p "$BAG_DIR"

set +u
source /opt/ros/noetic/setup.bash
source "$HOME/dog_nav_stack_demo/odin_ros_driver_main/devel/setup.bash" --extend 2>/dev/null || true
source "$HOME/dog_nav_stack_demo/autonomous_exploration_development_environment/devel/setup.bash" --extend 2>/dev/null || true
source "$HOME/dog_nav_stack_demo/iPlanner/devel/setup.bash" --extend 2>/dev/null || true
set -u

TOPICS=(
  /tf
  /tf_static

  # Data collector direct inputs
  /odin1/depth_img_competetion
  /odin1/image/undistorted
  /odin1/depth/camera_info
  /odin1/color/camera_info
  /registered_scan
  /state_estimation

  # Useful command/goal context for labeling/debugging
  /way_point
  /joy
  /go2/joy
)

if [[ "$INCLUDE_RAW_ODIN" == "true" ]]; then
  TOPICS+=(
    /odin1/odometry_highfreq
    /odin1/cloud_slam_i
    /odin1/cloud_slam
    /odin1/image
    /odin1/imu
  )
fi

if [[ "$INCLUDE_GO2" == "true" ]]; then
  TOPICS+=(
    /go2/odom
    /go2/sportmodestate
    /cmd_vel
  )
fi

EXISTING=()
MISSING=()
for topic in "${TOPICS[@]}"; do
  if rostopic type "$topic" >/dev/null 2>&1; then
    EXISTING+=("$topic")
  else
    MISSING+=("$topic")
  fi
done

if [[ ${#EXISTING[@]} -eq 0 ]]; then
  echo "[ERROR] No requested topics are currently available. Is roscore/Odin running?" >&2
  exit 2
fi

echo "[INFO] Bag dir: $BAG_DIR"
echo "[INFO] Bag prefix: $BAG_PREFIX"
echo "[INFO] Recording ${#EXISTING[@]} existing topics:"
printf '  %s\n' "${EXISTING[@]}"
if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "[WARN] Skipping ${#MISSING[@]} missing topics:"
  printf '  %s\n' "${MISSING[@]}"
fi

echo "[INFO] Stop recording with Ctrl-C."

COMPRESS_ARGS=()
case "$COMPRESS" in
  none|false|off) COMPRESS_ARGS=() ;;
  bz2) COMPRESS_ARGS=(--bz2) ;;
  lz4) COMPRESS_ARGS=(--lz4) ;;
  *) echo "[ERROR] Unsupported COMPRESS=$COMPRESS. Use bz2, lz4, or none." >&2; exit 3 ;;
esac

exec rosbag record \
  --split --size="$SPLIT_SIZE_MB" \
  "${COMPRESS_ARGS[@]}" \
  -O "$BAG_DIR/$BAG_PREFIX" \
  "${EXISTING[@]}"
