"""Gamepad + teach pendant. Start arm_driver and the FlexiGrasp pad node first."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_cfg = os.path.join(get_package_share_directory("open_manipulator_x6"), "config", "joy_ps4.yaml")
    cfg = LaunchConfiguration("config")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_cfg),
        Node(package="joy", executable="joy_node", name="joy_node",
             parameters=[{"deadzone": 0.05, "autorepeat_rate": 20.0}]),
        Node(package="open_manipulator_x6", executable="teach_pendant", name="teach_pendant",
             parameters=[cfg], output="screen"),
    ])