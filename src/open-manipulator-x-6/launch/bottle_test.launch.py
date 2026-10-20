"""
One command for a recorded playback run:
    ros2 launch open_manipulator_x6 bottle_test.launch.py sequence:=~/Open-Manipulator-X-6/bottle_sequence.yaml label:=bottle_1

Starts the session logger, a ros2 bag recording and the sequence runner, and shuts all three
down when the sequence finishes (or is stopped). arm_driver and the pad node must already run.
Outputs go to ~/flexigrasp_logs/: <stamp>_<label>.csv, <stamp>_<label>_steps.csv, <stamp>_<label>_bag/.
"""
import os
import time

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

TOPICS = ["/flexigrasp/left/frame", "/flexigrasp/right/frame", "/joint_states",
          "/arm_diagnostics", "/arm_status", "/sequence_step", "/arm_command"]


def _setup(context):
    seq = os.path.expanduser(context.launch_configurations["sequence"])
    label = context.launch_configurations["label"]
    out = os.path.expanduser(context.launch_configurations["out_dir"])
    os.makedirs(out, exist_ok=True)
    bag_dir = os.path.join(out, f"{time.strftime('%Y%m%d_%H%M%S')}_{label}_bag")
    topics = TOPICS + context.launch_configurations["extra_topics"].split()
    runner = Node(package="open_manipulator_x6", executable="sequence_runner", name="sequence_runner",
                  output="screen",
                  parameters=[{"sequence_file": seq, "label": label, "out_dir": out}])
    return [
        Node(package="open_manipulator_x6", executable="session_logger", name="session_logger",
             output="screen", parameters=[{"label": label, "out_dir": out}]),
        ExecuteProcess(cmd=["ros2", "bag", "record", "-o", bag_dir] + topics, output="screen"),
        runner,
        RegisterEventHandler(OnProcessExit(target_action=runner,
                                           on_exit=[EmitEvent(event=Shutdown(reason="sequence done"))])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("sequence", description="path to the sequence YAML"),
        DeclareLaunchArgument("label", default_value="bottle"),
        DeclareLaunchArgument("out_dir", default_value="~/flexigrasp_logs"),
        DeclareLaunchArgument("extra_topics", default_value="",
                              description="space-separated, e.g. /image_raw"),
        OpaqueFunction(function=_setup),
    ])