#!/usr/bin/env python3
"""
Gymnasium environment wrapping the Gazebo arm simulation.
Observation: [j1..j5 pos, j1..j5 vel, bag_x, bag_y, bag_z, tgt_x, tgt_y, tgt_z] (16 floats)
Action:      [dj1..dj5] joint position deltas, clipped to [-0.05, 0.05] rad per step
Reward:      shaped reward based on gripper-to-bag distance and bag-to-target distance
"""

import time
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, GetEntityState, SetEntityState
from gazebo_msgs.msg import EntityState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, Point, Quaternion

import threading


# Joint limits matching your URDF
JOINT_LIMITS = {
    'joint_1': (-3.14,  3.14),
    'joint_2': (-1.57,  1.57),
    'joint_3': (-1.57,  1.57),
    'joint_4': (-1.00,  2.00),
    'joint_5': (-3.14,  3.14),
}
JOINT_NAMES = list(JOINT_LIMITS.keys())

# Fixed target position (where the bag should be placed)
TARGET_POS = np.array([0.0, 0.25, 0.05])

# Bag spawn region (randomised each episode)
BAG_X_RANGE = (0.12, 0.18)
BAG_Y_RANGE = (-0.05, 0.05)
BAG_Z       = 0.06

# Step size for joint delta actions
ACTION_SCALE = 0.05   # radians per step

# Thresholds
PICKUP_HEIGHT    = 0.08   # bag z above this = considered lifted
SUCCESS_DIST     = 0.05   # bag within 5cm of target = success
MAX_STEPS        = 50    # episode length


class ROSBridge(Node):
    """Handles all ROS2 communication."""

    def __init__(self):
        super().__init__('rl_env_bridge')

        # Joint states subscriber
        self.joint_state = None
        self.js_lock = threading.Lock()
        self.create_subscription(
            JointState, '/joint_states', self._js_cb, 10)

        # Arm action client
        self.arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

        # Gripper action client
        self.gripper_client = ActionClient(
            self, FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory')

        # Gazebo services
        self.spawn_client      = self.create_client(SpawnEntity,    '/spawn_entity')
        self.delete_client     = self.create_client(DeleteEntity,   '/delete_entity')
        self.get_state_client  = self.create_client(GetEntityState, '/gazebo/get_entity_state')
        self.set_state_client  = self.create_client(SetEntityState, '/gazebo/set_entity_state')

    def _js_cb(self, msg):
        with self.js_lock:
            self.joint_state = msg

    def get_joint_state(self):
        with self.js_lock:
            return self.joint_state

    def send_arm_positions(self, positions, duration_sec=1):
        """Send joint positions and block until complete."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = list(positions)
        pt.time_from_start = Duration(sec=duration_sec)
        goal.trajectory.points = [pt]
        self.arm_client.wait_for_server()
        future = self.arm_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.02)
        gh = future.result()
        rf = gh.get_result_async()
        while not rf.done():
            time.sleep(0.02)

    def send_gripper(self, left, right, duration_sec=1):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['left_finger_joint', 'right_finger_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [left, right]
        pt.time_from_start = Duration(sec=duration_sec)
        goal.trajectory.points = [pt]
        self.gripper_client.wait_for_server()
        future = self.gripper_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.02)
        gh = future.result()
        rf = gh.get_result_async()
        while not rf.done():
            time.sleep(0.02)

    def get_bag_pose(self):
        """Returns bag [x, y, z] or None if unavailable."""
        req = GetEntityState.Request()
        req.name = 'paper_bag'
        req.reference_frame = 'world'
        future = self.get_state_client.call_async(req)
        timeout = time.time() + 2.0
        while not future.done():
            if time.time() > timeout:
                return None
            time.sleep(0.02)
        result = future.result()
        if result is None or not result.success:
            return None
        p = result.state.pose.position
        return np.array([p.x, p.y, p.z])

    def teleport_bag(self, x, y, z):
        """Move bag to a new position instantly."""
        req = SetEntityState.Request()
        state = EntityState()
        state.name = 'paper_bag'
        state.pose = Pose(
            position=Point(x=x, y=y, z=z),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
        state.reference_frame = 'world'
        req.state = state
        future = self.set_state_client.call_async(req)
        timeout = time.time() + 2.0
        while not future.done():
            if time.time() > timeout:
                return
            time.sleep(0.02)


class GazeboArmEnv(gym.Env):
    """
    Gymnasium environment for the 5-DOF arm pick and place task.

    Observation (16,):
        [j1..j5 positions, j1..j5 velocities, bag_x, bag_y, bag_z,
         target_x, target_y, target_z]

    Action (5,):
        Joint position deltas [dj1..dj5] in [-1, 1], scaled by ACTION_SCALE
    """

    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()

        # Observation: 5 pos + 5 vel + 3 bag + 3 target
        obs_low  = np.array(
            [l for l, _ in JOINT_LIMITS.values()] +   # joint pos low
            [-5.0] * 5 +                               # joint vel low
            [-2.0, -2.0,  0.0] +                       # bag xyz low
            [-2.0, -2.0,  0.0],                        # target xyz low
            dtype=np.float32)
        obs_high = np.array(
            [h for _, h in JOINT_LIMITS.values()] +
            [ 5.0] * 5 +
            [ 2.0,  2.0,  2.0] +
            [ 2.0,  2.0,  2.0],
            dtype=np.float32)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        # ROS2 init
        if not rclpy.ok():
            rclpy.init()
        self.ros = ROSBridge()
        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self.ros,), daemon=True)
        self._spin_thread.start()

        # Wait for joint states to arrive
        print('Waiting for joint states...')
        while self.ros.get_joint_state() is None:
            time.sleep(0.1)
        print('Joint states received.')

        self._current_joints = np.zeros(5, dtype=np.float32)
        self._step_count      = 0
        self._bag_pos         = np.array([0.15, 0.0, BAG_Z], dtype=np.float32)
        self._prev_bag_dist   = None
        self._episode_reward  = 0.0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count     = 0
        self._episode_reward = 0.0

        # Home the arm and open gripper
        home = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.ros.send_arm_positions(home, duration_sec=2)
        self.ros.send_gripper(0.0, 0.0, duration_sec=1)

        # Randomise bag spawn position
        bx = random.uniform(*BAG_X_RANGE)
        by = random.uniform(*BAG_Y_RANGE)
        self.ros.teleport_bag(bx, by, BAG_Z)
        time.sleep(0.3)  # let physics settle

        self._bag_pos = np.array([bx, by, BAG_Z], dtype=np.float32)
        self._current_joints = np.zeros(5, dtype=np.float32)
        self._prev_bag_dist  = np.linalg.norm(self._bag_pos - TARGET_POS)

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action):
        self._step_count += 1

        # Apply joint deltas, clip to limits
        new_joints = self._current_joints + action * ACTION_SCALE
        for i, name in enumerate(JOINT_NAMES):
            lo, hi = JOINT_LIMITS[name]
            new_joints[i] = float(np.clip(new_joints[i], lo, hi))

        # Send to controller — short duration for responsive training
        self.ros.send_arm_positions(new_joints.tolist(), duration_sec=0)
        self._current_joints = new_joints

        # Get bag pose
        bag = self.ros.get_bag_pose()
        if bag is not None:
            self._bag_pos = bag.astype(np.float32)

        # --- Reward shaping ---
        bag_dist   = float(np.linalg.norm(self._bag_pos - TARGET_POS))
        dist_delta = self._prev_bag_dist - bag_dist  # positive = moved closer
        self._prev_bag_dist = bag_dist

        reward = 0.0

        # 1. Reward for moving bag closer to target
        reward += dist_delta * 10.0

        # 2. Bonus if bag is lifted (z above pickup threshold)
        if self._bag_pos[2] > PICKUP_HEIGHT:
            reward += 0.5

        # 3. Large bonus for reaching target
        success = bag_dist < SUCCESS_DIST
        if success:
            reward += 50.0

        # 4. Small time penalty to encourage efficiency
        reward -= 0.01

        self._episode_reward += reward

        # Termination conditions
        terminated = success
        truncated  = self._step_count >= MAX_STEPS

        if terminated:
            print(f'SUCCESS! Episode reward: {self._episode_reward:.2f}')
        elif truncated:
            print(f'Truncated. Episode reward: {self._episode_reward:.2f}')

        obs  = self._get_obs()
        info = {
            'bag_dist':      bag_dist,
            'episode_reward': self._episode_reward,
            'step':          self._step_count,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _get_obs(self):
        js = self.ros.get_joint_state()
        pos = np.zeros(5, dtype=np.float32)
        vel = np.zeros(5, dtype=np.float32)

        if js is not None:
            name_to_idx = {n: i for i, n in enumerate(js.name)}
            for i, name in enumerate(JOINT_NAMES):
                if name in name_to_idx:
                    idx = name_to_idx[name]
                    if idx < len(js.position):
                        pos[i] = float(js.position[idx])
                    if idx < len(js.velocity):
                        vel[i] = float(js.velocity[idx])

        return np.concatenate([
            pos,
            vel,
            self._bag_pos,
            TARGET_POS.astype(np.float32),
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    def close(self):
        rclpy.shutdown()
