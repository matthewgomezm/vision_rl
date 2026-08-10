import os

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer

from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from config.go2_config import EnvironmentConfig
from environment.go2_env import UnitreeGo2Env

# Per-keypress increments and clamps for [vx, vy, wz]. Ranges mirror CommandConfig.
CMD_STEP = jnp.array([0.25, 0.25, 0.25])
CMD_MAX = jnp.array([1.5, 1.0, 1.2])


def load_policy(policy_path: str, env: UnitreeGo2Env):
    """Rebuild the training-time network and bind the saved params to it."""
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key='state',
        value_obs_key='privileged_state',
    )
    ppo_network = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    params = model.load_params(policy_path)
    inference_fn = make_inference_fn(params, deterministic=True)
    return jax.jit(inference_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True,
                        help='Path to a saved policy, e.g. policies/20260716-142821')
    parser.add_argument('--scene', type=str, default=EnvironmentConfig.filename)
    args = parser.parse_args()

    env = UnitreeGo2Env(environment_config=EnvironmentConfig(filename=args.scene))
    policy_fn = load_policy(args.policy, env)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)

    # Keyboard command, mutated by the callback and read each frame.
    command = {'value': jnp.zeros(3), 'reset': False}

    def key_callback(keycode):
        key = chr(keycode) if 0 < keycode < 0x110000 else ''
        delta = jnp.zeros(3)
        if key == 'W':
            delta = jnp.array([CMD_STEP[0], 0.0, 0.0])
        elif key == 'S':
            delta = jnp.array([-CMD_STEP[0], 0.0, 0.0])
        elif key == 'Q':
            delta = jnp.array([0.0, CMD_STEP[1], 0.0])
        elif key == 'E':
            delta = jnp.array([0.0, -CMD_STEP[1], 0.0])
        elif key == 'A':
            delta = jnp.array([0.0, 0.0, CMD_STEP[2]])
        elif key == 'D':
            delta = jnp.array([0.0, 0.0, -CMD_STEP[2]])
        elif keycode == 32:  # space
            command['value'] = jnp.zeros(3)
            return
        elif key == 'R':
            command['reset'] = True
            return
        command['value'] = jnp.clip(command['value'] + delta, -CMD_MAX, CMD_MAX)

    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)
    dt = env.dt

    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=key_callback
    ) as viewer:
        act_rng = jax.random.PRNGKey(1)
        last_cmd = None
        while viewer.is_running():
            step_start = time.time()

            if command['reset']:
                rng, sub = jax.random.split(rng)
                state = jit_reset(sub)
                command['reset'] = False

            # Pin our keyboard command into the env so step() won't resample it.
            state.info['command'] = command['value']
            state.info['steps_until_next_command'] = jnp.int32(1_000_000)

            act_rng, sub = jax.random.split(act_rng)
            action, _ = policy_fn(state.obs, sub)
            state = jit_step(state, action)

            # Mirror MJX state into the CPU model for rendering.
            mj_data.qpos[:] = np.asarray(state.data.qpos)
            mj_data.qvel[:] = np.asarray(state.data.qvel)
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            cmd = tuple(round(float(c), 2) for c in command['value'])
            if cmd != last_cmd:
                print(f'command [vx, vy, wz] = {cmd}')
                last_cmd = cmd

            # Real-time pacing (50 Hz control).
            wait = dt - (time.time() - step_start)
            if wait > 0:
                time.sleep(wait)


if __name__ == '__main__':
    main()
