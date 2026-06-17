#!/usr/bin/env python3
"""
Run a trained PPO agent.

Usage:
    python3 run_agent.py ppo_arm_final
    python3 run_agent.py checkpoints/ppo_arm_10000_steps
"""

import sys
import time
from stable_baselines3 import PPO
from gazebo_env import GazeboArmEnv

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'ppo_arm_final'
    print(f'Loading model from {model_path}...')
    model = PPO.load(model_path)

    env = GazeboArmEnv()
    episodes = 10

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        print(f'\n--- Episode {ep+1}/{episodes} ---')
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            print(f'  step {steps:3d} | bag_dist: {info["bag_dist"]:.3f}m | '
                  f'reward: {reward:6.3f}', end='\r')

        print(f'\n  Total reward: {total_reward:.2f} in {steps} steps')

    env.close()

if __name__ == '__main__':
    main()
