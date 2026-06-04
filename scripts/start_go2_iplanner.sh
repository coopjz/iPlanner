#!/usr/bin/env bash
set -euo pipefail

# One-command startup for Odin + Go2 bridge + iPlanner-viz/AEE pathFollower.
# Hardware/SDK nodes are launched through their native workspace launch files;
# go2_odin_iplanner*.launch remains iPlanner/AEE-only.

NETWORK_INTERFACE="${NETWORK_INTERFACE:-eth0}"
START_ODIN="${START_ODIN:-auto}"
START_GO2_CONTROL="${START_GO2_CONTROL:-auto}"
START_GO2_STATE="${START_GO2_STATE:-auto}"
START_RVIZ="${START_RVIZ:-auto}"
USE_IPLANNER_VIZ="${USE_IPLANNER_VIZ:-true}"
ODIN_WS="${ODIN_WS:-$HOME/dog_nav_stack_demo/odin_ros_driver_main}"
GO2_WS="${GO2_WS:-$HOME/dog_nav_stack_demo/go2_sdk/ros_ws}"
ODIN_LAUNCH_FILE="${ODIN_LAUNCH_FILE:-$ODIN_WS/src/odin_ros_driver/launch_ROS1/odin1_ros1.launch}"
GO2_TWIST_LAUNCH_FILE="${GO2_TWIST_LAUNCH_FILE:-$GO2_WS/src/go2_ros_bridge/launch/go2_twist_bridge.launch}"
GO2_STATE_LAUNCH_FILE="${GO2_STATE_LAUNCH_FILE:-$GO2_WS/src/go2_ros_bridge/launch/go2_state_bridge.launch}"
ROS_MASTER_PORT="${ROS_MASTER_PORT:-11311}"
LOG_DIR="${LOG_DIR:-/tmp/go2_iplanner_logs}"
ODIN_LAUNCH_PID=""
GO2_LAUNCH_PID=""
GO2_STATE_LAUNCH_PID=""
IP_LAUNCH_PID=""

mkdir -p "$LOG_DIR"

source_ros_env() {
  # ROS/catkin setup files may read unset variables. Keep strict mode for this
  # script, but temporarily disable nounset while sourcing external setup files.
  set +u
  source /opt/ros/noetic/setup.bash
  source "$HOME/dog_nav_stack_demo/odin_ros_driver_main/devel/setup.bash" --extend
  source "$HOME/dog_nav_stack_demo/autonomous_exploration_development_environment/devel/setup.bash" --extend
  source "$HOME/dog_nav_stack_demo/go2_sdk/ros_ws/devel/setup.bash" --extend
  source "$HOME/dog_nav_stack_demo/iPlanner/devel/setup.bash" --extend
  set -u
}

source_ros_env

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:${ROS_MASTER_PORT}}"

check_master_port() {
  local line owner
  line="$(ss -ltnp 2>/dev/null | grep ":${ROS_MASTER_PORT} " || true)"
  [[ -z "$line" ]] && return 0

  owner="$(sed -n 's/.*users:(("\([^"]*\)".*/\1/p' <<<"$line" | head -1)"
  if [[ "$owner" == "rosmaster" || "$owner" == "python" || "$owner" == "roscore" ]]; then
    echo "[INFO] ROS master port ${ROS_MASTER_PORT} already has a ROS-looking listener: $line"
    return 0
  fi

  echo "[ERROR] Port ${ROS_MASTER_PORT} is occupied by a non-ROS process:" >&2
  echo "$line" >&2
  echo "Free it first, e.g. inspect with: ss -ltnp | grep ${ROS_MASTER_PORT}" >&2
  echo "If it is VS Code/code-server port forwarding, close that forwarded port." >&2
  exit 1
}

configure_rviz() {
  case "$START_RVIZ" in
    auto|AUTO)
      if [[ -n "${DISPLAY:-}" ]]; then
        START_RVIZ=true
      elif [[ -S /tmp/.X11-unix/X0 ]]; then
        export DISPLAY=:0
        START_RVIZ=true
        echo "[INFO] DISPLAY was empty; using local display DISPLAY=:0 for RViz."
      else
        START_RVIZ=false
        echo "[WARN] No DISPLAY found; RViz will not be started. Set START_RVIZ=true DISPLAY=:0 if running on the robot desktop."
      fi
      ;;
    true|false) ;;
    *) echo "[ERROR] START_RVIZ must be auto, true, or false; got '$START_RVIZ'" >&2; exit 1 ;;
  esac
}

rosnode_exists() {
  rosnode list 2>/dev/null | grep -qx "$1"
}

topic_has_type() {
  [[ -n "$(rostopic type "$1" 2>/dev/null || true)" ]]
}

configure_iplanner_launch() {
  case "$USE_IPLANNER_VIZ" in
    true)
      IPLANNER_LAUNCH="go2_odin_iplanner_viz.launch"
      ;;
    false)
      IPLANNER_LAUNCH="go2_odin_iplanner.launch"
      ;;
    *) echo "[ERROR] USE_IPLANNER_VIZ must be true or false; got '$USE_IPLANNER_VIZ'" >&2; exit 1 ;;
  esac
}

validate_launch_files() {
  if [[ "$START_ODIN" == "true" && ! -d "$ODIN_WS" ]]; then
    echo "[ERROR] ODIN_WS does not exist: $ODIN_WS" >&2
    exit 1
  fi
  if [[ "$START_ODIN" == "true" && ! -f "$ODIN_LAUNCH_FILE" ]]; then
    echo "[ERROR] ODIN_LAUNCH_FILE does not exist: $ODIN_LAUNCH_FILE" >&2
    exit 1
  fi
  if [[ "$START_GO2_CONTROL" == "true" || "$START_GO2_STATE" == "true" ]]; then
    if [[ ! -d "$GO2_WS" ]]; then
      echo "[ERROR] GO2_WS does not exist: $GO2_WS" >&2
      exit 1
    fi
  fi
  if [[ "$START_GO2_CONTROL" == "true" && ! -f "$GO2_TWIST_LAUNCH_FILE" ]]; then
    echo "[ERROR] GO2_TWIST_LAUNCH_FILE does not exist: $GO2_TWIST_LAUNCH_FILE" >&2
    exit 1
  fi
  if [[ "$START_GO2_STATE" == "true" && ! -f "$GO2_STATE_LAUNCH_FILE" ]]; then
    echo "[ERROR] GO2_STATE_LAUNCH_FILE does not exist: $GO2_STATE_LAUNCH_FILE" >&2
    exit 1
  fi
}

configure_start_flags() {
  case "$START_ODIN" in
    auto|AUTO)
      if topic_has_type /odin1/depth_img_competetion; then
        START_ODIN=false
        echo "[INFO] Existing Odin depth topic detected; not starting Odin native launch."
      else
        START_ODIN=true
      fi
      ;;
    true|false) ;;
    *) echo "[ERROR] START_ODIN must be auto, true, or false; got '$START_ODIN'" >&2; exit 1 ;;
  esac

  case "$START_GO2_STATE" in
    auto|AUTO)
      if rosnode_exists /go2_state_bridge_node; then
        START_GO2_STATE=false
        echo "[INFO] Existing /go2_state_bridge_node detected; not starting go2_state_bridge again."
      else
        START_GO2_STATE=true
      fi
      ;;
    true|false) ;;
    *) echo "[ERROR] START_GO2_STATE must be auto, true, or false; got '$START_GO2_STATE'" >&2; exit 1 ;;
  esac

  case "$START_GO2_CONTROL" in
    auto|AUTO)
      if rosnode_exists /go2_twist_bridge_node; then
        START_GO2_CONTROL=false
        echo "[INFO] Existing /go2_twist_bridge_node detected; not starting go2_twist_bridge again."
      else
        START_GO2_CONTROL=true
      fi
      ;;
    true|false) ;;
    *) echo "[ERROR] START_GO2_CONTROL must be auto, true, or false; got '$START_GO2_CONTROL'" >&2; exit 1 ;;
  esac
}

cleanup() {
  echo "[INFO] Shutting down launched roslaunch processes..."
  [[ -n "${IP_LAUNCH_PID:-}" ]] && kill -INT "$IP_LAUNCH_PID" 2>/dev/null || true
  [[ -n "${GO2_LAUNCH_PID:-}" ]] && kill -INT "$GO2_LAUNCH_PID" 2>/dev/null || true
  [[ -n "${GO2_STATE_LAUNCH_PID:-}" ]] && kill -INT "$GO2_STATE_LAUNCH_PID" 2>/dev/null || true
  [[ -n "${ODIN_LAUNCH_PID:-}" ]] && kill -INT "$ODIN_LAUNCH_PID" 2>/dev/null || true
  wait "${IP_LAUNCH_PID:-0}" 2>/dev/null || true
  wait "${GO2_LAUNCH_PID:-0}" 2>/dev/null || true
  wait "${GO2_STATE_LAUNCH_PID:-0}" 2>/dev/null || true
  wait "${ODIN_LAUNCH_PID:-0}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

check_master_port
configure_rviz
configure_start_flags
validate_launch_files
configure_iplanner_launch

echo "[INFO] ROS_MASTER_URI=$ROS_MASTER_URI"
echo "[INFO] START_RVIZ=$START_RVIZ DISPLAY=${DISPLAY:-}"
echo "[INFO] START_ODIN=$START_ODIN START_GO2_CONTROL=$START_GO2_CONTROL START_GO2_STATE=$START_GO2_STATE"
echo "[INFO] ODIN_WS=$ODIN_WS"
echo "[INFO] ODIN_LAUNCH_FILE=$ODIN_LAUNCH_FILE"
echo "[INFO] GO2_WS=$GO2_WS"
echo "[INFO] GO2_TWIST_LAUNCH_FILE=$GO2_TWIST_LAUNCH_FILE"
echo "[INFO] GO2_STATE_LAUNCH_FILE=$GO2_STATE_LAUNCH_FILE"
echo "[INFO] IPLANNER_LAUNCH=$IPLANNER_LAUNCH"
echo "[INFO] Logs: $LOG_DIR"

if [[ "$START_ODIN" == "true" ]]; then
  echo "[INFO] Starting Odin sensor from workspace: $ODIN_WS"
  echo "[INFO] Odin launch file: $ODIN_LAUNCH_FILE"
  (cd "$ODIN_WS" && roslaunch "$ODIN_LAUNCH_FILE") >"$LOG_DIR/odin_ros_driver.log" 2>&1 &
  ODIN_LAUNCH_PID=$!

  sleep 3
  if ! kill -0 "$ODIN_LAUNCH_PID" 2>/dev/null; then
    echo "[ERROR] Odin launch exited early. Tail log:" >&2
    tail -80 "$LOG_DIR/odin_ros_driver.log" >&2 || true
    exit 1
  fi
else
  echo "[INFO] Skipping Odin launch."
fi

if [[ "$START_GO2_STATE" == "true" ]]; then
  echo "[INFO] Starting Go2 state bridge from workspace: $GO2_WS"
  echo "[INFO] Go2 state launch file: $GO2_STATE_LAUNCH_FILE"
  (cd "$GO2_WS" && roslaunch "$GO2_STATE_LAUNCH_FILE" \
    network_interface:="$NETWORK_INTERFACE") >"$LOG_DIR/go2_state_bridge.log" 2>&1 &
  GO2_STATE_LAUNCH_PID=$!

  sleep 2
  if ! kill -0 "$GO2_STATE_LAUNCH_PID" 2>/dev/null; then
    echo "[ERROR] Go2 state bridge launch exited early. Tail log:" >&2
    tail -80 "$LOG_DIR/go2_state_bridge.log" >&2 || true
    exit 1
  fi
else
  echo "[INFO] Skipping Go2 state bridge launch."
fi

if [[ "$START_GO2_CONTROL" == "true" ]]; then
  echo "[INFO] Starting Go2 twist bridge from workspace: $GO2_WS"
  echo "[INFO] Go2 launch file: $GO2_TWIST_LAUNCH_FILE"
  (cd "$GO2_WS" && roslaunch "$GO2_TWIST_LAUNCH_FILE" \
    network_interface:="$NETWORK_INTERFACE") >"$LOG_DIR/go2_twist_bridge.log" 2>&1 &
  GO2_LAUNCH_PID=$!

  sleep 2
  if ! kill -0 "$GO2_LAUNCH_PID" 2>/dev/null; then
    echo "[ERROR] Go2 twist bridge launch exited early. Tail log:" >&2
    tail -80 "$LOG_DIR/go2_twist_bridge.log" >&2 || true
    exit 1
  fi
else
  echo "[INFO] Skipping Go2 twist bridge launch."
fi

echo "[INFO] Starting iPlanner control launch..."
roslaunch iplanner_node "$IPLANNER_LAUNCH" \
  start_rviz:="$START_RVIZ" >"$LOG_DIR/go2_odin_iplanner.log" 2>&1 &
IP_LAUNCH_PID=$!

sleep 5
if ! kill -0 "$IP_LAUNCH_PID" 2>/dev/null; then
  echo "[ERROR] iPlanner launch exited early. Tail log:" >&2
  tail -100 "$LOG_DIR/go2_odin_iplanner.log" >&2 || true
  exit 1
fi

echo "[INFO] Started. PIDs: odin=${ODIN_LAUNCH_PID:-none} go2_state=${GO2_STATE_LAUNCH_PID:-none} go2_twist=${GO2_LAUNCH_PID:-none} iplanner=$IP_LAUNCH_PID"
echo "[INFO] Follow logs with: tail -f $LOG_DIR/odin_ros_driver.log $LOG_DIR/go2_state_bridge.log $LOG_DIR/go2_twist_bridge.log $LOG_DIR/go2_odin_iplanner.log"
if [[ "$START_RVIZ" == "true" ]]; then echo "[INFO] RViz should start from $IPLANNER_LAUNCH."; fi
echo "[INFO] Press Ctrl-C here to stop both launches."

WAIT_PIDS=()
[[ -n "$ODIN_LAUNCH_PID" ]] && WAIT_PIDS+=("$ODIN_LAUNCH_PID")
[[ -n "$GO2_STATE_LAUNCH_PID" ]] && WAIT_PIDS+=("$GO2_STATE_LAUNCH_PID")
[[ -n "$GO2_LAUNCH_PID" ]] && WAIT_PIDS+=("$GO2_LAUNCH_PID")
WAIT_PIDS+=("$IP_LAUNCH_PID")
wait "${WAIT_PIDS[@]}"
