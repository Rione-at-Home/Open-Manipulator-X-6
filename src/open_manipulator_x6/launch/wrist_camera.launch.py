"""
Wrist camera (Elecom) with its calibration, optionally with the camera-frame marker checker.

    ros2 launch open_manipulator_x6 wrist_camera.launch.py
    ros2 launch open_manipulator_x6 wrist_camera.launch.py checker:=true
    ros2 launch open_manipulator_x6 wrist_camera.launch.py video_device:=/dev/v4l/by-id/<...>-video-index0

Settings (device, 1280x720, frame name, calibration file) are in config/wrist_camera.yaml; set
video_device there once. Publishes /wrist_cam/image_raw and /wrist_cam/camera_info.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _camera(context):
    cfg_path = LaunchConfiguration("config").perform(context)
    with open(cfg_path) as f:
        params = yaml.safe_load(f)["wrist_cam"]["ros__parameters"]
    device = LaunchConfiguration("video_device").perform(context) or params["video_device"]
    if "REPLACE-WITH" in device:
        raise RuntimeError("Set video_device in config/wrist_camera.yaml (see `ls /dev/v4l/by-id/`), "
                           "or pass video_device:=... on the command line.")
    if not os.path.exists(device):
        raise RuntimeError(f"Wrist camera not found at {device}. Is it plugged in? "
                           f"Check `ls /dev/v4l/by-id/`.")
    url = params["camera_info_url"]
    if url.startswith("file://~"):
        url = "file://" + os.path.expanduser(url[len("file://"):])
    if url.startswith("file://") and not os.path.exists(url[len("file://"):]):
        print(f"[wrist_camera] warning: no calibration file at {url[len('file://'):]} "
              f"(run camera_calibrator); camera_info will be empty")
    params = dict(params, video_device=device, camera_info_url=url)
    actions = [Node(package="v4l2_camera", executable="v4l2_camera_node", name="v4l2_camera",
                    namespace="wrist_cam", parameters=[params], output="screen")]
    if LaunchConfiguration("checker").perform(context).lower() in ("true", "1", "yes"):
        # marker sizes etc. from aruco_markers.yaml (keyed to aruco_detector), then camera-frame settings
        share = get_package_share_directory("open_manipulator_x6")
        with open(os.path.join(share, "config", "aruco_markers.yaml")) as f:
            aruco = yaml.safe_load(f)["aruco_detector"]["ros__parameters"]
        aruco.update({"frame": "camera", "image_topic": "/wrist_cam/image_raw",
                      "camera_info_topic": "/wrist_cam/camera_info", "log_file": ""})
        actions.append(Node(package="open_manipulator_x6", executable="aruco_detector", name="wrist_aruco",
                            parameters=[aruco], output="screen"))
    return actions


def generate_launch_description():
    share = get_package_share_directory("open_manipulator_x6")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=os.path.join(share, "config", "wrist_camera.yaml")),
        DeclareLaunchArgument("video_device", default_value="",
                              description="override the device in the config file"),
        DeclareLaunchArgument("checker", default_value="false",
                              description="also start the marker checker in camera frame"),
        OpaqueFunction(function=_camera),
    ])