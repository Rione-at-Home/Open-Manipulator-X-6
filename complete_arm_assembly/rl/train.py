#!/usr/bin/env python3
"""
Train a PPO agent to pick and place the paper bag.

Usage:
    source /opt/ros/humble/setup.bash
    source ~/ros2_ws/install/setup.bash
    python3 train.py

Checkpoints saved to: ./checkpoints/
Tensorboard logs:     ./logs/
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback)
from stable_baselines3.common.monitor import Monitor
from gazebo_env import GazeboArmEnv


class EpisodeLogCallback(BaseCallback):
    """Prints episode stats to terminal."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                r = info['episode']['r']
                l = info['episode']['l']
                self.episode_rewards.append(r)
                mean = np.mean(self.episode_rewards[-20:])
                print(f'  Episode {len(self.episode_rewards):4d} | '
                      f'reward: {r:8.2f} | steps: {l:4d} | '
                      f'mean(20): {mean:8.2f}')
        return True


def main():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs',        exist_ok=True)

    print('Initialising Gazebo environment...')
    env = Monitor(GazeboArmEnv())

    # PPO hyperparameters tuned for a slow real-time Gazebo environment
    model = PPO(
        policy          = 'MlpPolicy',
        env             = env,
        learning_rate   = 3e-4,
        n_steps         = 128,       # steps per rollout (small = more updates)
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,      # encourages exploration
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        tensorboard_log = './logs/',
        verbose         = 1,
        policy_kwargs   = dict(
            net_arch = [256, 256],   # two hidden layers
        ),
    )

    callbacks = [
        EpisodeLogCallback(),
        CheckpointCallback(
            save_freq    = 2000,
            save_path    = './checkpoints/',
            name_prefix  = 'ppo_arm',
        ),
    ]

    print('Starting training — Gazebo must be running with the arm spawned.')
    print('Press Ctrl+C to stop and save.\n')

    try:
        model.learn(
            total_timesteps  = 20_000,
            callback         = callbacks,
            tb_log_name      = 'PPO_arm',
            reset_num_timesteps = True,
        )
    except KeyboardInterrupt:
        print('\nTraining interrupted — saving model...')

    model.save('ppo_arm_final')
    print('Model saved to ppo_arm_final.zip')
    env.close()


if __name__ == '__main__':
    main()
