import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, RegisterEventHandler,
                            SetEnvironmentVariable)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('complete_arm_assembly')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    xacro_file = os.path.join(pkg_share, 'urdf', 'complete_arm_assembly_urdf.xacro')
    urdf_path = os.path.join(pkg_share, 'urdf', 'complete_arm_assembly.urdf')

    urdf_content = subprocess.check_output(['xacro', xacro_file]).decode('utf-8')
    with open(urdf_path, 'w') as f:
        f.write(urdf_content)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        )
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf_content}],
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-file', urdf_path, '-entity', 'complete_arm_assembly'],
        output='screen',
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    return LaunchDescription([
        # Tell gazebo_ros2_control to load URDF from file, not param server.
        # This env var is checked by the plugin before it tries the param route.
        SetEnvironmentVariable('GAZEBO_ROS2_CONTROL_URDF_FILE', urdf_path),
        gazebo,
        robot_state_publisher_node,
        spawn_entity,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[joint_state_broadcaster_spawner],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[arm_controller_spawner, gripper_controller_spawner],
            )
        ),
    ])