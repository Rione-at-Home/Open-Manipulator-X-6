"""
RViz display of the X6.

    ros2 launch open_manipulator_x6_description display.launch.py              # sliders
    ros2 launch open_manipulator_x6_description display.launch.py mode:=live   # follow the real arm

mode:=gui   joint_state_publisher_gui sliders drive the model (no hardware).
mode:=live  the model follows arm_driver's /joint_states. gripper_joint_bridge turns the driver's
            'gripper' angle into finger travel and republishes on /joint_states_urdf, which
            robot_state_publisher listens to. Start arm_driver separately.
rviz:=false starts only robot_state_publisher (and the sliders or bridge), e.g. for Isaac Sim tests.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    share = get_package_share_directory("open_manipulator_x6_description")
    mode = LaunchConfiguration("mode").perform(context).strip().lower()
    if mode not in ("gui", "live"):
        raise RuntimeError(f"mode must be 'gui' or 'live', not '{mode}'")
    with open(os.path.join(share, "urdf", "open_manipulator_x6.urdf")) as f:
        description = f.read()
    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
               parameters=[{"robot_description": description}],
               remappings=[("/joint_states", "/joint_states_urdf")] if mode == "live" else [])
    nodes = [rsp]
    if mode == "gui":
        nodes.append(Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui",
                          output="screen"))
    else:
        nodes.append(Node(package="open_manipulator_x6", executable="gripper_joint_bridge",
                          name="gripper_joint_bridge", output="screen",
                          parameters=[os.path.join(share, "config", "gripper_bridge.yaml")]))
    if LaunchConfiguration("rviz").perform(context).lower() in ("true", "1", "yes"):
        nodes.append(Node(package="rviz2", executable="rviz2", output="screen",
                          arguments=["-d", os.path.join(share, "rviz", "display.rviz")]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="gui", description="gui (sliders) or live (real arm)"),
        DeclareLaunchArgument("rviz", default_value="true"),
        OpaqueFunction(function=_nodes),
    ])
