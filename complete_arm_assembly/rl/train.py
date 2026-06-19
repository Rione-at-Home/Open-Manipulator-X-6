#!/usr/bin/env python3
import sys
import os
import numpy as np
sys.stdout.reconfigure(line_buffering=True)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from gazebo_env import GazeboArmEnv


class EpisodeLogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                r = info['episode']['r']
                l = info['episode']['l']
                self.episode_rewards.append(r)
                mean = np.mean(self.episode_rewards[-20:])
                print(f'  Episode {len(self.episode_rewards):4d} | '
                      f'reward: {r:8.2f} | steps: {l:4d} | '
                      f'mean(20): {mean:8.2f}', flush=True)
        return True


def main():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    print('Initialising Gazebo environment...')
    env = Monitor(GazeboArmEnv())

    model = PPO(
        policy          = 'MlpPolicy',
        env             = env,
        learning_rate   = 3e-4,
        n_steps         = 2048,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        tensorboard_log = './logs/',
        verbose         = 1,
        device          = 'cpu',  # CPU is faster for MLP policies
        policy_kwargs   = dict(net_arch=[256, 256]),
    )

    callbacks = [
        EpisodeLogCallback(),
        CheckpointCallback(
            save_freq   = 2000,
            save_path   = './checkpoints/',
            name_prefix = 'ppo_arm',
        ),
    ]

    print('Starting training — Gazebo must be running with the arm spawned.')
    print('Press Ctrl+C to stop and save.\n')

    try:
        model.learn(
            total_timesteps     = 20_000,
            callback            = callbacks,
            tb_log_name         = 'PPO_arm',
            reset_num_timesteps = True,
        )
    except KeyboardInterrupt:
        print('\nTraining interrupted — saving model...')

    model.save('ppo_arm_final')
    print('Model saved to ppo_arm_final.zip')
    env.close()


if __name__ == '__main__':
    main()
