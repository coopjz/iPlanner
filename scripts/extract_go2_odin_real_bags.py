#!/usr/bin/env python3
"""Extract Go2/Odin real-world ROS1 bags into iPlanner CollectedData format.

This is an offline replacement for data_collect_node.py when bags already contain
Odin's deskewed/registered point cloud.  The point cloud timestamp is the sample
clock; odom and depth are matched by nearest-neighbour timestamps.
"""

import argparse
import bisect
import glob
import json
import math
import os
import shutil
import sys
import collections
import collections.abc
from collections import defaultdict

# ros_numpy still references old collections aliases on Python 3.10+.
for _name in ("Sequence", "Mapping", "MutableMapping", "Iterable"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d
import rosbag
import ros_numpy
from rosbag.bag import ROSBagException, ROSBagFormatException, ROSBagUnindexedException
from scipy.spatial.transform import Rotation as R


DEPTH_TOPIC = "/odin1/depth_img_competetion"
COLOR_TOPIC = "/odin1/image/undistorted"
ODOM_TOPIC = "/odin1/odometry_highfreq"
PRIMARY_CLOUD_TOPIC = "/odin1/cloud_slam_i"
FALLBACK_CLOUD_TOPIC = "/odin1/cloud_slam"


@dataclass(frozen=True)
class Match:
    sample_id: int
    cloud_idx: int
    odom_idx: int
    depth_idx: int
    color_idx: Optional[int]
    cloud_stamp: float
    odom_dt: float
    depth_dt: float
    color_dt: Optional[float]


@dataclass(frozen=True)
class ImageTransform:
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    target_w: int
    target_h: int
    scale_x: float
    scale_y: float


@dataclass(frozen=True)
class SelfFilterBox:
    enabled: bool
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


def parse_projection_matrix(path: str) -> np.ndarray:
    text = open(path, "r").read().strip()
    cleaned = text.translate(str.maketrans({"[": " ", "]": " ", "(": " ", ")": " "}))
    elems = np.fromstring(cleaned, dtype=float, sep=",")
    if elems.size == 12:
        return elems.reshape(3, 4)
    if elems.size == 16:
        return elems.reshape(4, 4)[:3, :4]
    raise ValueError("Expected 12 or 16 projection/intrinsic values in %s, got %d" % (path, elems.size))


def write_projection_matrix(path: str, P: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write(str(tuple(float(v) for v in P.reshape(-1))) + "\n")


def compute_center_crop(src_w: int, src_h: int, target_w: int, target_h: int) -> ImageTransform:
    target_aspect = float(target_w) / float(target_h)
    src_aspect = float(src_w) / float(src_h)
    if src_aspect > target_aspect:
        crop_h = src_h
        crop_w = int(round(crop_h * target_aspect))
        crop_x = (src_w - crop_w) // 2
        crop_y = 0
    else:
        crop_w = src_w
        crop_h = int(round(crop_w / target_aspect))
        crop_x = 0
        crop_y = (src_h - crop_h) // 2
    return ImageTransform(
        crop_x=crop_x,
        crop_y=crop_y,
        crop_w=crop_w,
        crop_h=crop_h,
        target_w=target_w,
        target_h=target_h,
        scale_x=float(target_w) / float(crop_w),
        scale_y=float(target_h) / float(crop_h),
    )


def transform_projection(P_raw: np.ndarray, transform: ImageTransform) -> np.ndarray:
    P = np.array(P_raw, dtype=float, copy=True)
    P[0, 2] -= transform.crop_x
    P[1, 2] -= transform.crop_y
    P[0, :] *= transform.scale_x
    P[1, :] *= transform.scale_y
    return P


def ensure_empty_env(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError("Output env exists; rerun with --overwrite: %s" % path)
        shutil.rmtree(path)
    os.makedirs(os.path.join(path, "depth"), exist_ok=True)
    os.makedirs(os.path.join(path, "camera"), exist_ok=True)
    os.makedirs(os.path.join(path, "scan"), exist_ok=True)


def nearest_index_and_delta(stamp: float, stamps: Sequence[float]) -> Tuple[Optional[int], Optional[float]]:
    if not stamps:
        return None, None
    pos = bisect.bisect_left(stamps, stamp)
    candidates: List[Tuple[int, float]] = []
    if pos < len(stamps):
        candidates.append((pos, stamps[pos] - stamp))
    if pos > 0:
        candidates.append((pos - 1, stamps[pos - 1] - stamp))
    idx, delta = min(candidates, key=lambda item: abs(item[1]))
    return idx, delta


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * pct / 100.0))
    return float(arr[idx])


def summarize_dts(values: Sequence[float]) -> Dict[str, Optional[float]]:
    abs_values = [abs(float(v)) for v in values]
    if not abs_values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(abs_values),
        "mean": float(np.mean(abs_values)),
        "p50": percentile(abs_values, 50),
        "p95": percentile(abs_values, 95),
        "p99": percentile(abs_values, 99),
        "max": float(max(abs_values)),
    }


def odom_to_list(msg) -> List[float]:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return [float(p.x), float(p.y), float(p.z), float(q.x), float(q.y), float(q.z), float(q.w)]


def crop_resize_color(image: np.ndarray, transform: ImageTransform) -> np.ndarray:
    cropped = image[transform.crop_y:transform.crop_y + transform.crop_h,
                    transform.crop_x:transform.crop_x + transform.crop_w]
    return cv2.resize(cropped, (transform.target_w, transform.target_h), interpolation=cv2.INTER_AREA)


def crop_resize_depth_to_uint16(depth_m: np.ndarray, transform: ImageTransform) -> np.ndarray:
    depth = depth_m[transform.crop_y:transform.crop_y + transform.crop_h,
                    transform.crop_x:transform.crop_x + transform.crop_w].astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    weighted = np.where(valid, depth, 0.0).astype(np.float32)
    coverage = valid.astype(np.float32)
    resized_sum = cv2.resize(weighted, (transform.target_w, transform.target_h), interpolation=cv2.INTER_AREA)
    resized_cov = cv2.resize(coverage, (transform.target_w, transform.target_h), interpolation=cv2.INTER_AREA)
    resized = np.zeros_like(resized_sum, dtype=np.float32)
    np.divide(resized_sum, resized_cov, out=resized, where=resized_cov > 1e-6)
    resized[~np.isfinite(resized)] = 0.0
    resized[resized < 0.0] = 0.0
    return np.clip(np.round(resized * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def ros_image_to_numpy(msg) -> np.ndarray:
    """Decode common sensor_msgs/Image encodings without cv_bridge/ros_numpy."""
    encoding = msg.encoding.lower()
    channels = 1
    scale_to_meters = 1.0
    color_order = None

    if encoding in ("32fc1", "32fc"):
        dtype = np.dtype(np.float32)
    elif encoding in ("16uc1", "mono16"):
        dtype = np.dtype(np.uint16)
        scale_to_meters = 1.0 / 1000.0
    elif encoding in ("8uc1", "mono8"):
        dtype = np.dtype(np.uint8)
    elif encoding in ("bgr8", "rgb8"):
        dtype = np.dtype(np.uint8)
        channels = 3
        color_order = encoding
    elif encoding in ("bgra8", "rgba8"):
        dtype = np.dtype(np.uint8)
        channels = 4
        color_order = encoding
    else:
        raise ValueError("Unsupported image encoding: %s" % msg.encoding)

    if msg.is_bigendian and dtype.itemsize > 1:
        dtype = dtype.newbyteorder(">")
    elif dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<")

    row_items = msg.step // dtype.itemsize
    if channels > 1:
        row_pixels = row_items // channels
        array = np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, row_pixels, channels))[:, :msg.width, :]
        if color_order == "rgb8":
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        elif color_order == "rgba8":
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
        elif color_order == "bgra8":
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
        return np.array(array, copy=True)

    array = np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, row_items))[:, :msg.width]
    array = np.array(array, copy=True)
    if scale_to_meters != 1.0:
        array = array.astype(np.float32) * scale_to_meters
    return array


def filter_self_points(points: np.ndarray, odom: Sequence[float], self_filter: SelfFilterBox) -> Tuple[np.ndarray, int]:
    if (not self_filter.enabled) or points.size == 0:
        return points, 0

    odom_array = np.asarray(odom, dtype=np.float64)
    rotation = R.from_quat(odom_array[3:7]).as_matrix()
    translation = odom_array[:3]
    local = (rotation.T @ (points - translation).T).T
    in_self_box = (
        (local[:, 0] >= self_filter.x_min) & (local[:, 0] <= self_filter.x_max) &
        (local[:, 1] >= self_filter.y_min) & (local[:, 1] <= self_filter.y_max) &
        (local[:, 2] >= self_filter.z_min) & (local[:, 2] <= self_filter.z_max)
    )
    return points[~in_self_box], int(np.count_nonzero(in_self_box))


def save_cloud_xyz(path: str, msg, odom: Sequence[float], self_filter: SelfFilterBox) -> Tuple[int, int]:
    points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=True)
    points = points.astype(np.float64, copy=False)
    points, removed_self_points = filter_self_points(points, odom, self_filter)

    pcd = o3d.geometry.PointCloud()
    if points.size > 0:
        pcd.points = o3d.utility.Vector3dVector(points)
    ok = o3d.io.write_point_cloud(path, pcd)
    if not ok:
        raise RuntimeError("Open3D failed to write point cloud: %s" % path)
    return int(points.shape[0]), removed_self_points


def open_bag(path: str):
    try:
        return rosbag.Bag(path)
    except (ROSBagUnindexedException, ROSBagFormatException, ROSBagException):
        return rosbag.Bag(path, allow_unindexed=True)


def collect_lightweight_index(bag_path: str, topics: Sequence[str]) -> Tuple[Dict[str, List[float]], List[List[float]], Dict[str, str]]:
    stamps: Dict[str, List[float]] = {topic: [] for topic in topics}
    odoms: List[List[float]] = []
    frames: Dict[str, str] = {}
    with open_bag(bag_path) as bag:
        for topic, msg, _ in bag.read_messages(topics=topics):
            header = getattr(msg, "header", None)
            if header is None:
                continue
            stamps[topic].append(float(header.stamp.to_sec()))
            if topic not in frames:
                frames[topic] = getattr(header, "frame_id", "")
            if topic == ODOM_TOPIC:
                odoms.append(odom_to_list(msg))
    return stamps, odoms, frames


def build_matches(
    cloud_stamps: Sequence[float],
    odom_stamps: Sequence[float],
    depth_stamps: Sequence[float],
    color_stamps: Sequence[float],
    odom_slop: float,
    depth_slop: float,
    color_slop: float,
    max_frames: Optional[int],
    frame_stride: int,
) -> Tuple[List[Match], Dict[str, int]]:
    matches: List[Match] = []
    drops = {
        "no_odom": 0,
        "odom_slop": 0,
        "no_depth": 0,
        "depth_slop": 0,
        "stride": 0,
    }
    accepted_seen = 0
    for cloud_idx, stamp in enumerate(cloud_stamps):
        odom_idx, odom_dt = nearest_index_and_delta(stamp, odom_stamps)
        if odom_idx is None or odom_dt is None:
            drops["no_odom"] += 1
            continue
        if abs(odom_dt) > odom_slop:
            drops["odom_slop"] += 1
            continue

        depth_idx, depth_dt = nearest_index_and_delta(stamp, depth_stamps)
        if depth_idx is None or depth_dt is None:
            drops["no_depth"] += 1
            continue
        if abs(depth_dt) > depth_slop:
            drops["depth_slop"] += 1
            continue

        if accepted_seen % frame_stride != 0:
            accepted_seen += 1
            drops["stride"] += 1
            continue
        accepted_seen += 1

        color_idx, color_dt = nearest_index_and_delta(stamp, color_stamps)
        if color_dt is not None and abs(color_dt) > color_slop:
            color_idx = None
            color_dt = None

        matches.append(Match(
            sample_id=len(matches),
            cloud_idx=cloud_idx,
            odom_idx=odom_idx,
            depth_idx=depth_idx,
            color_idx=color_idx,
            cloud_stamp=float(stamp),
            odom_dt=float(odom_dt),
            depth_dt=float(depth_dt),
            color_dt=None if color_dt is None else float(color_dt),
        ))
        if max_frames is not None and len(matches) >= max_frames:
            break
    return matches, drops


def write_metadata(env_path: str, P_scaled: np.ndarray, camera_extrinsic_path: str) -> None:
    write_projection_matrix(os.path.join(env_path, "depth_intrinsic.txt"), P_scaled)
    write_projection_matrix(os.path.join(env_path, "color_intrinsic.txt"), P_scaled)
    shutil.copyfile(camera_extrinsic_path, os.path.join(env_path, "camera_extrinsic.txt"))
    with open(os.path.join(env_path, "scan_extrinsic.txt"), "w") as f:
        f.write(str([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) + "\n")


def process_selected_messages(
    bag_path: str,
    env_path: str,
    cloud_topic: str,
    matches: Sequence[Match],
    odoms: Sequence[Sequence[float]],
    transform: ImageTransform,
    self_filter: SelfFilterBox,
) -> Dict[str, int]:
    cloud_to_samples = defaultdict(list)
    depth_to_samples = defaultdict(list)
    color_to_samples = defaultdict(list)
    for match in matches:
        cloud_to_samples[match.cloud_idx].append(match.sample_id)
        depth_to_samples[match.depth_idx].append(match.sample_id)
        if match.color_idx is not None:
            color_to_samples[match.color_idx].append(match.sample_id)

    saved = {"cloud": 0, "depth": 0, "color": 0, "black_color": 0, "removed_self_points": 0}
    cloud_points: Dict[int, int] = {}
    removed_self_points_by_sample: Dict[int, int] = {}
    color_written = set()
    counters = defaultdict(int)

    wanted_topics = [cloud_topic, DEPTH_TOPIC, COLOR_TOPIC]
    with open_bag(bag_path) as bag:
        for topic, msg, _ in bag.read_messages(topics=wanted_topics):
            idx = counters[topic]
            counters[topic] += 1
            if topic == cloud_topic and idx in cloud_to_samples:
                for sample_id in cloud_to_samples[idx]:
                    count, removed_self_points = save_cloud_xyz(
                        os.path.join(env_path, "scan", "%d.ply" % sample_id),
                        msg,
                        odoms[matches[sample_id].odom_idx],
                        self_filter)
                    cloud_points[sample_id] = count
                    removed_self_points_by_sample[sample_id] = removed_self_points
                    saved["removed_self_points"] += removed_self_points
                    saved["cloud"] += 1
            elif topic == DEPTH_TOPIC and idx in depth_to_samples:
                depth = ros_image_to_numpy(msg)
                for sample_id in depth_to_samples[idx]:
                    depth_u16 = crop_resize_depth_to_uint16(depth, transform)
                    cv2.imwrite(os.path.join(env_path, "depth", "%d.png" % sample_id), depth_u16)
                    saved["depth"] += 1
            elif topic == COLOR_TOPIC and idx in color_to_samples:
                color = ros_image_to_numpy(msg)
                if color.ndim == 2:
                    color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
                for sample_id in color_to_samples[idx]:
                    color_small = crop_resize_color(color, transform)
                    cv2.imwrite(os.path.join(env_path, "camera", "%d.png" % sample_id), color_small)
                    color_written.add(sample_id)
                    saved["color"] += 1

    black = np.zeros((transform.target_h, transform.target_w, 3), dtype=np.uint8)
    for match in matches:
        if match.sample_id not in color_written:
            cv2.imwrite(os.path.join(env_path, "camera", "%d.png" % match.sample_id), black)
            saved["black_color"] += 1

    with open(os.path.join(env_path, "odom_ground_truth.txt"), "w") as f:
        for match in matches:
            f.write(str([float(v) for v in odoms[match.odom_idx]]) + "\n")

    saved["min_cloud_points"] = int(min(cloud_points.values())) if cloud_points else 0
    saved["max_cloud_points"] = int(max(cloud_points.values())) if cloud_points else 0
    saved["mean_removed_self_points"] = float(np.mean(list(removed_self_points_by_sample.values()))) if removed_self_points_by_sample else 0.0
    saved["max_removed_self_points"] = int(max(removed_self_points_by_sample.values())) if removed_self_points_by_sample else 0
    return saved


def extract_bag(args, bag_path: str, env_name: str, P_raw: np.ndarray) -> Tuple[bool, Dict]:
    env_path = os.path.join(args.output_root, env_name)
    report = {
        "bag_path": bag_path,
        "env_name": env_name,
        "env_path": env_path,
        "success": False,
    }

    topics = [PRIMARY_CLOUD_TOPIC, FALLBACK_CLOUD_TOPIC, DEPTH_TOPIC, COLOR_TOPIC, ODOM_TOPIC]
    try:
        stamps, odoms, frames = collect_lightweight_index(bag_path, topics)
    except Exception as exc:
        report.update({"error": "%s: %s" % (type(exc).__name__, exc)})
        return False, report

    cloud_topic = PRIMARY_CLOUD_TOPIC if len(stamps[PRIMARY_CLOUD_TOPIC]) > 0 else FALLBACK_CLOUD_TOPIC
    cloud_stamps = stamps[cloud_topic]
    report.update({
        "topic_counts": {topic: len(stamps.get(topic, [])) for topic in topics},
        "frames": frames,
        "cloud_topic": cloud_topic,
    })

    if not cloud_stamps or not stamps[DEPTH_TOPIC] or not stamps[ODOM_TOPIC]:
        report.update({"error": "Missing required cloud/depth/odom topic messages."})
        return False, report

    matches, drops = build_matches(
        cloud_stamps=cloud_stamps,
        odom_stamps=stamps[ODOM_TOPIC],
        depth_stamps=stamps[DEPTH_TOPIC],
        color_stamps=stamps[COLOR_TOPIC],
        odom_slop=args.odom_slop,
        depth_slop=args.depth_slop,
        color_slop=args.color_slop,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
    )
    report.update({
        "drop_counts": drops,
        "matched_samples": len(matches),
        "cloud_odom_dt_abs_sec": summarize_dts([m.odom_dt for m in matches]),
        "cloud_depth_dt_abs_sec": summarize_dts([m.depth_dt for m in matches]),
        "cloud_color_dt_abs_sec": summarize_dts([m.color_dt for m in matches if m.color_dt is not None]),
    })

    if len(matches) == 0:
        report.update({"error": "No samples passed nearest-neighbour sync thresholds."})
        return False, report

    ensure_empty_env(env_path, args.overwrite)

    transform = compute_center_crop(
        src_w=args.source_width,
        src_h=args.source_height,
        target_w=args.target_width,
        target_h=args.target_height,
    )
    P_scaled = transform_projection(P_raw, transform)
    write_metadata(env_path, P_scaled, args.camera_extrinsic)
    self_filter = SelfFilterBox(
        enabled=not args.disable_self_filter,
        x_min=args.self_filter_x_min,
        x_max=args.self_filter_x_max,
        y_min=args.self_filter_y_min,
        y_max=args.self_filter_y_max,
        z_min=args.self_filter_z_min,
        z_max=args.self_filter_z_max,
    )

    try:
        saved = process_selected_messages(bag_path, env_path, cloud_topic, matches, odoms, transform, self_filter)
    except Exception as exc:
        report.update({"error": "%s: %s" % (type(exc).__name__, exc)})
        return False, report

    report.update({
        "success": True,
        "saved_counts": saved,
        "image_transform": transform.__dict__,
        "projection_matrix": [float(v) for v in P_scaled.reshape(-1)],
        "self_filter_box_sensor_frame": self_filter.__dict__,
        "sync_policy": {
            "clock": cloud_topic,
            "odom_topic": ODOM_TOPIC,
            "depth_topic": DEPTH_TOPIC,
            "color_topic": COLOR_TOPIC,
            "odom_slop_sec": args.odom_slop,
            "depth_slop_sec": args.depth_slop,
            "color_slop_sec": args.color_slop,
            "nearest_neighbour": True,
            "interpolation": False,
        },
    })

    with open(os.path.join(env_path, "extract_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return True, report


def write_list(path: str, env_names: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for env_name in env_names:
            f.write(env_name + "\n")


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", default="/media/cooper/XiangruT7/go2_odin_finetune")
    parser.add_argument("--bag-glob", default="*.bag")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--env-prefix", default="go2_odin_real")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--collect-list", default=None)
    parser.add_argument("--training-list", default=None)
    parser.add_argument("--intrinsic", required=True, help="Reference Odin 1600x1296 projection/intrinsic txt.")
    parser.add_argument("--camera-extrinsic", required=True, help="Reference camera_extrinsic.txt to copy.")
    parser.add_argument("--source-width", type=int, default=1600)
    parser.add_argument("--source-height", type=int, default=1296)
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument("--target-height", type=int, default=360)
    parser.add_argument("--odom-slop", type=float, default=0.005)
    parser.add_argument("--depth-slop", type=float, default=0.020)
    parser.add_argument("--color-slop", type=float, default=0.250)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--disable-self-filter", action="store_true",
                        help="Disable head-mounted robot self/near-field point filter.")
    parser.add_argument("--self-filter-x-min", type=float, default=-0.20)
    parser.add_argument("--self-filter-x-max", type=float, default=0.85)
    parser.add_argument("--self-filter-y-min", type=float, default=-0.32)
    parser.add_argument("--self-filter-y-max", type=float, default=0.32)
    parser.add_argument("--self-filter-z-min", type=float, default=-0.30)
    parser.add_argument("--self-filter-z-max", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-bad-bag", action="store_true")
    args = parser.parse_args(argv)
    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.output_root, exist_ok=True)
    P_raw = parse_projection_matrix(args.intrinsic)

    bag_paths = sorted(glob.glob(os.path.join(args.bag_dir, args.bag_glob)))
    if not bag_paths:
        print("No bags matched: %s" % os.path.join(args.bag_dir, args.bag_glob), file=sys.stderr)
        return 2

    reports = []
    successful_envs: List[str] = []
    for offset, bag_path in enumerate(bag_paths):
        env_name = "%s_%03d" % (args.env_prefix, args.start_index + offset)
        print("=== Extracting %s -> %s ===" % (os.path.basename(bag_path), env_name), flush=True)
        ok, report = extract_bag(args, bag_path, env_name, P_raw)
        reports.append(report)
        if ok:
            successful_envs.append(env_name)
            print("saved %d samples; odom p95 %.6fs; depth p95 %.6fs" % (
                report["matched_samples"],
                report["cloud_odom_dt_abs_sec"]["p95"] or math.nan,
                report["cloud_depth_dt_abs_sec"]["p95"] or math.nan,
            ), flush=True)
        else:
            print("skip %s: %s" % (env_name, report.get("error", "unknown error")), file=sys.stderr, flush=True)
            if args.fail_on_bad_bag:
                break

    summary = {
        "bag_dir": args.bag_dir,
        "bag_count": len(bag_paths),
        "successful_envs": successful_envs,
        "failed_envs": [r["env_name"] for r in reports if not r.get("success")],
        "reports": reports,
    }
    summary_path = os.path.join(args.output_root, "%s_extract_summary.json" % args.env_prefix)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    if args.collect_list:
        write_list(args.collect_list, successful_envs)
    if args.training_list:
        write_list(args.training_list, successful_envs)

    print("=== Extraction summary ===")
    print("successful envs: %d/%d" % (len(successful_envs), len(bag_paths)))
    print("summary: %s" % summary_path)
    if args.collect_list:
        print("collect list: %s" % args.collect_list)
    if args.training_list:
        print("training list: %s" % args.training_list)
    return 0 if successful_envs else 3


if __name__ == "__main__":
    sys.exit(main())
