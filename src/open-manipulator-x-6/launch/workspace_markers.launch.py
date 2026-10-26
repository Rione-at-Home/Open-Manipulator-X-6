"""
World camera (C920) + workspace marker checker.

    ros2 launch open_manipulator_x6 workspace_markers.launch.py video_device:=/dev/v4l/by-id/<C920>-video-index0

The camera calibration is read from ~/.ros/camera_info/world_cam.yaml (written by the
calibration tool's COMMIT button, see the README).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("open_manipulator_x6"), "config", "aruco_markers.yaml")
    info_url = "file://" + os.path.expanduser("~/.ros/camera_info/world_cam.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("video_device", default_value="/dev/video0"),
        DeclareLaunchArgument("config", default_value=cfg),
        Node(package="v4l2_camera", executable="v4l2_camera_node", name="world_cam",
             namespace="world_cam",
             parameters=[{"video_device": LaunchConfiguration("video_device"),
                          "image_size": [1280, 720],
                          "camera_frame_id": "world_cam",
                          "camera_info_url": info_url}]),
        Node(package="open_manipulator_x6", executable="aruco_detector", name="aruco_detector",
             parameters=[LaunchConfiguration("config")], output="screen"),
    ])