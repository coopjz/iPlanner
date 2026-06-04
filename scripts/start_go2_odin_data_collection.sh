#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-go2_odin_001}"
START_LOAM_INTERFACE="${START_LOAM_INTERFACE:-true}"
START_CAMERA_INFO="${START_CAMERA_INFO:-true}"
LOG_DIR="${LOG_DIR:-/home/unitree/dog_nav_stack_demo/iPlanner/logs/go2_odin_data_collection}"
mkdir -p "$LOG_DIR"

set +u
source /opt/ros/noetic/setup.bash
source "$HOME/dog_nav_stack_demo/odin_ros_driver_main/devel/setup.bash" --extend
source "$HOME/dog_nav_stack_demo/autonomous_exploration_development_environment/devel/setup.bash" --extend
source "$HOME/dog_nav_stack_demo/iPlanner/devel/setup.bash" --extend
set -u

if [[ "$START_LOAM_INTERFACE" == "auto" ]]; then
  if [[ -n "$(rostopic type /state_estimation 2>/dev/null || true)" && -n "$(rostopic type /registered_scan 2>/dev/null || true)" ]]; then
    START_LOAM_INTERFACE=false
  else
    START_LOAM_INTERFACE=true
  fi
fi

echo "[INFO] ENV_NAME=$ENV_NAME"
echo "[INFO] START_LOAM_INTERFACE=$START_LOAM_INTERFACE START_CAMERA_INFO=$START_CAMERA_INFO"
echo "[INFO] Logs: $LOG_DIR/go2_odin_data_collector.log"
roslaunch iplanner_node go2_odin_data_collector.launch \
  env_name:="$ENV_NAME" \
  start_loam_interface:="$START_LOAM_INTERFACE" \
  start_camera_info:="$START_CAMERA_INFO" \
  >"$LOG_DIR/go2_odin_data_collector.log" 2>&1
