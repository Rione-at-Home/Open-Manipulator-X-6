import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('open_manipulator_x6'),
        'config',
        'motors.yaml'
    )

    return LaunchDescription([
        Node(
            package='open_manipulator_x6',
            executable='arm_driver',
            name='arm_driver',
            output='screen',
            parameters=[config]
        )
    ])