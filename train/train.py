import os

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

import argparse
import functools
from datetime import datetime
from pathlib import Path

import jax
import wandb

from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from mujoco_playground import wrapper

from config.go2_config import EnvironmentConfig
from environment.go2_env import UnitreeGo2Env

# Change this to log runs under a different Weights & Biases project.
WANDB_PROJECT = 'unitree-go2-vision-rl'


def ppo_params(num_timesteps: int, num_envs: int) -> dict:
    return {
        'num_timesteps': num_timesteps,
        'num_envs': num_envs,
        'episode_length': 1000,
        'num_evals': 10,
        'batch_size': 256,
        'unroll_length': 20,
        'num_minibatches': 32,
        'num_updates_per_batch': 4,
        'discounting': 0.97,
        'learning_rate': 3e-4,
        'entropy_cost': 1e-2,
        'reward_scaling': 1.0,
        'max_grad_norm': 1.0,
        'action_repeat': 1,
        'normalize_observations': True,
        'deterministic_eval': True,
    }


def make_progress_fn(run=None):
    def progress(num_steps, metrics):
        reward = metrics.get('eval/episode_reward', float('nan'))
        length = metrics.get('eval/avg_episode_length', float('nan'))
        print(f'[{num_steps:>12,}] reward={reward:8.3f}  ep_len={length:7.1f}')
        if run is not None:
            run.log({k: float(v) for k, v in metrics.items()}, step=int(num_steps))

    return progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-timesteps', type=int, default=200_000_000)
    parser.add_argument('--num-envs', type=int, default=2048)
    parser.add_argument('--scene', type=str, default=EnvironmentConfig.filename)
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--notes', type=str, default='')
    args = parser.parse_args()

    save_path = args.save_path or f'policies/{datetime.now():%Y%m%d-%H%M%S}'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    env = UnitreeGo2Env(
        environment_config=EnvironmentConfig(filename=args.scene),
    )

    params = ppo_params(args.num_timesteps, args.num_envs)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key='state',
        value_obs_key='privileged_state',
    )

    print(f'training {args.scene} for {args.num_timesteps:,} steps '
          f'across {args.num_envs} envs -> {save_path}')

    wandb.login()
    with wandb.init(
        project=WANDB_PROJECT,
        name=Path(save_path).name,
        notes=args.notes,
        config={
            'ppo_params': params,
            'reward_config': env.reward_config,
            'command_config': env.command_config,
            'env_config': env.environment_config,
            'scene': args.scene,
            'save_path': save_path,
        },
    ) as run:
        train_fn = functools.partial(
            ppo.train,
            **params,
            network_factory=network_factory,
            progress_fn=make_progress_fn(run),
            seed=0,
        )
        make_inference_fn, trained_params, _ = train_fn(
            environment=env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
        del make_inference_fn

        model.save_params(save_path, trained_params)
        print(f'saved policy to {save_path}')


if __name__ == '__main__':
    main()
