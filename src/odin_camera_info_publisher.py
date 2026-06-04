#!/usr/bin/env python3
"""Publish CameraInfo for Odin depth/color streams from odin_ros_driver calib.yaml.

The upstream Odin ROS1 driver publishes Image topics but no standard CameraInfo topic.
iPlanner's data collector only needs the projection matrix P, so this bridge
publishes a pinhole approximation from the Odin FishPoly calibration intrinsics.
"""

import os
import rospy
import yaml
from sensor_msgs.msg import CameraInfo


def _load_camera(calib_file, camera_key):
    with open(os.path.expanduser(calib_file), 'r') as f:
        calib = yaml.safe_load(f)
    if camera_key not in calib:
        raise KeyError("camera key '%s' not found in %s" % (camera_key, calib_file))
    cam = calib[camera_key]
    width = int(cam.get('image_width', 0))
    height = int(cam.get('image_height', 0))
    fx = float(cam.get('A11', 0.0))
    skew = float(cam.get('A12', 0.0))
    fy = float(cam.get('A22', 0.0))
    cx = float(cam.get('u0', 0.0))
    cy = float(cam.get('v0', 0.0))
    distortion = [
        float(cam.get('k2', 0.0)),
        float(cam.get('k3', 0.0)),
        float(cam.get('k4', 0.0)),
        float(cam.get('k5', 0.0)),
        float(cam.get('k6', 0.0)),
        float(cam.get('k7', 0.0)),
        float(cam.get('p1', 0.0)),
        float(cam.get('p2', 0.0)),
    ]
    return width, height, fx, skew, fy, cx, cy, distortion


def _make_info(width, height, fx, skew, fy, cx, cy, distortion, frame_id, stamp):
    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.width = width
    msg.height = height
    msg.distortion_model = 'plumb_bob'
    msg.D = distortion
    msg.K = [fx, skew, cx,
             0.0, fy, cy,
             0.0, 0.0, 1.0]
    msg.R = [1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0]
    msg.P = [fx, skew, cx, 0.0,
             0.0, fy, cy, 0.0,
             0.0, 0.0, 1.0, 0.0]
    return msg


def main():
    rospy.init_node('odin_camera_info_publisher', anonymous=False)
    calib_file = rospy.get_param('~calib_file', os.path.expanduser('~/dog_nav_stack_demo/odin_ros_driver_main/src/odin_ros_driver/config/calib.yaml'))
    camera_key = rospy.get_param('~camera_key', 'cam_0')
    frame_id = rospy.get_param('~frame_id', 'camera')
    depth_info_topic = rospy.get_param('~depth_info_topic', '/odin1/depth/camera_info')
    color_info_topic = rospy.get_param('~color_info_topic', '/odin1/color/camera_info')
    publish_rate = float(rospy.get_param('~publish_rate', 5.0))

    width, height, fx, skew, fy, cx, cy, distortion = _load_camera(calib_file, camera_key)
    if width <= 0 or height <= 0 or fx <= 0.0 or fy <= 0.0:
        raise RuntimeError('Invalid Odin camera calibration loaded from %s' % calib_file)

    depth_pub = rospy.Publisher(depth_info_topic, CameraInfo, queue_size=1, latch=True)
    color_pub = rospy.Publisher(color_info_topic, CameraInfo, queue_size=1, latch=True)
    rospy.loginfo('Publishing Odin CameraInfo from %s[%s] to %s and %s, frame_id=%s, size=%dx%d',
                  calib_file, camera_key, depth_info_topic, color_info_topic, frame_id, width, height)

    rate = rospy.Rate(publish_rate)
    while not rospy.is_shutdown():
        stamp = rospy.Time.now()
        info = _make_info(width, height, fx, skew, fy, cx, cy, distortion, frame_id, stamp)
        depth_pub.publish(info)
        color_pub.publish(info)
        rate.sleep()


if __name__ == '__main__':
    main()
