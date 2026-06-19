#!/usr/bin/env python3
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, Point, Quaternion

import threading

JOINT_LIMITS = {
    'joint_1': (-3.14,  3.14),
    'joint_2': (-1.57,  1.57),
    'joint_3': (-1.57,  1.57),
    'joint_4': (-1.00,  2.00),
    'joint_5': (-3.14,  3.14),
}
JOINT_NAMES = list(JOINT_LIMITS.keys())

TARGET_POS   = np.array([0.0, 0.25, 0.05])
BAG_X_RANGE  = (0.12, 0.18)
BAG_Y_RANGE  = (-0.05, 0.05)
BAG_Z        = 0.06
ACTION_SCALE = 0.05
PICKUP_HEIGHT = 0.08
SUCCESS_DIST  = 0.05
MAX_STEPS     = 50


class ROSBridge(Node):
    def __init__(self):
        super().__init__('rl_env_bridge')

        self.joint_state = None
        self.js_lock = threading.Lock()
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)

        # Track bag via topic — no broken service needed
        self.model_states = None
        self.ms_lock = threading.Lock()
        self.create_subscription(ModelStates, '/gazebo/model_states', self._ms_cb, 10)

        self.arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(
            self, FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory')

        # Only set_entity_state for teleporting bag
        self.set_state_client = self.create_client(
            SetEntityState, '/gazebo/set_entity_state')

    def _js_cb(self, msg):
        with self.js_lock:
            self.joint_state = msg

    def _ms_cb(self, msg):
        with self.ms_lock:
            self.model_states = msg

    def get_joint_state(self):
        with self.js_lock:
            return self.joint_state

    def get_bag_pose(self):
        """Get bag position from /gazebo/model_states topic."""
        with self.ms_lock:
            ms = self.model_states
        if ms is None:
            return None
        try:
            idx = ms.name.index('paper_bag')
            p = ms.pose[idx].position
            return np.array([p.x, p.y, p.z], dtype=np.float32)
        except ValueError:
            return None

    def teleport_bag(self, x, y, z):
        """Teleport bag to new position."""
        if not self.set_state_client.wait_for_service(timeout_sec=1.0):
            print('[WARN] set_entity_state not available, skipping teleport')
            return
        req = SetEntityState.Request()
        state = EntityState()
        state.name = 'paper_bag'
        state.pose = Pose(
            position=Point(x=float(x), y=float(y), z=float(z)),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0
        state.reference_frame = 'world'
        req.state = state
        future = self.set_state_client.call_async(req)
        timeout = time.time() + 2.0
        while not future.done():
            if time.time() > timeout:
                return
            time.sleep(0.02)

    def send_arm_positions(self, positions, duration_sec=0):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = list(positions)
        pt.time_from_start = Duration(sec=max(1, duration_sec))
        goal.trajectory.points = [pt]
        self.arm_client.wait_for_server()
        # Fire and forget for speed during training
        self.arm_client.send_goal_async(goal)
        time.sleep(0.3)

    def send_gripper(self, left, right, duration_sec=1):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['left_finger_joint', 'right_finger_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [left, right]
        pt.time_from_start = Duration(sec=max(1, duration_sec))
        goal.trajectory.points = [pt]
        self.gripper_client.wait_for_server()
        self.gripper_client.send_goal_async(goal)
        time.sleep(0.2)


class GazeboArmEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()

        # Observation: 5 pos + 5 vel + 3 bag + 3 target + 1 gripper = 17
        obs_low = np.array(
            [l for l, _ in JOINT_LIMITS.values()] +
            [-5.0] * 5 +
            [-2.0, -2.0, 0.0] +
            [-2.0, -2.0, 0.0] +
            [0.0],
            dtype=np.float32)
        obs_high = np.array(
            [h for _, h in JOINT_LIMITS.values()] +
            [5.0] * 5 +
            [2.0, 2.0, 2.0] +
            [2.0, 2.0, 2.0] +
            [1.0],
            dtype=np.float32)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        # Action: 5 joint deltas + 1 gripper
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        if not rclpy.ok():
            rclpy.init()
        self.ros = ROSBridge()
        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self.ros,), daemon=True)
        self._spin_thread.start()

        print('Waiting for joint states...')
        while self.ros.get_joint_state() is None:
            time.sleep(0.1)
        print('Joint states received.')

        self._current_joints = np.zeros(5, dtype=np.float32)
        self._gripper_state  = 0.0
        self._step_count     = 0
        self._bag_pos        = np.array([0.15, 0.0, BAG_Z], dtype=np.float32)
        self._prev_bag_dist  = None
        self._episode_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count     = 0
        self._episode_reward = 0.0
        self._gripper_state  = 0.0

        # Home arm and open gripper
        home = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.ros.send_arm_positions(home, duration_sec=2)
        self.ros.send_gripper(0.0, 0.0, duration_sec=1)
        time.sleep(1.0)

        # Randomise bag position
        bx = random.uniform(*BAG_X_RANGE)
        by = random.uniform(*BAG_Y_RANGE)
        self.ros.teleport_bag(bx, by, BAG_Z)
        time.sleep(0.5)

        self._bag_pos = np.array([bx, by, BAG_Z], dtype=np.float32)
        self._current_joints = np.zeros(5, dtype=np.float32)
        self._prev_bag_dist = np.linalg.norm(self._bag_pos - TARGET_POS)

        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1

        # Apply joint deltas
        new_joints = self._current_joints + action[:5] * ACTION_SCALE
        for i, name in enumerate(JOINT_NAMES):
            lo, hi = JOINT_LIMITS[name]
            new_joints[i] = float(np.clip(new_joints[i], lo, hi))

        self.ros.send_arm_positions(new_joints.tolist(), duration_sec=1)
        self._current_joints = new_joints

        # Gripper action
        gripper_cmd = float(action[5])
        if gripper_cmd > 0.3:
            self.ros.send_gripper(0.019, -0.019, duration_sec=1)
            self._gripper_state = 1.0
        elif gripper_cmd < -0.3:
            self.ros.send_gripper(0.0, 0.0, duration_sec=1)
            self._gripper_state = 0.0

        # Get bag pose from topic
        bag = self.ros.get_bag_pose()
        if bag is not None:
            self._bag_pos = bag

        # Reward
        bag_dist   = float(np.linalg.norm(self._bag_pos - TARGET_POS))
        dist_delta = self._prev_bag_dist - bag_dist
        self._prev_bag_dist = bag_dist

        reward = dist_delta * 10.0
        if self._gripper_state == 1.0 and self._bag_pos[2] < PICKUP_HEIGHT + 0.02:
            reward += 1.0
        if self._bag_pos[2] > PICKUP_HEIGHT:
            reward += 2.0
        success = bag_dist < SUCCESS_DIST
        if success:
            reward += 50.0
        reward -= 0.01

        self._episode_reward += reward
        terminated = success
        truncated  = self._step_count >= MAX_STEPS

        if terminated:
            print(f'SUCCESS! Episode reward: {self._episode_reward:.2f}')
        elif truncated:
            print(f'Truncated. Episode reward: {self._episode_reward:.2f}')

        return self._get_obs(), reward, terminated, truncated, {
            'bag_dist': bag_dist,
            'episode_reward': self._episode_reward,
            'step': self._step_count,
        }

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
            pos, vel, self._bag_pos,
            TARGET_POS.astype(np.float32),
            [self._gripper_state],
        ]).astype(np.float32)

    def close(self):
        rclpy.shutdown()
