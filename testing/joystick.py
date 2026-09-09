import os

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

os.environ['SDL_VIDEODRIVER'] = 'dummy' 


import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer
import pygame

from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from config.go2_config import EnvironmentConfig
from environment.go2_env import UnitreeGo2Env
from environment.terrain import apply_random_terrain

CMD_MAX = jnp.array([1.5, 1.0, 1.2])
AXIS_FORWARD = 1          # left stick vertical    -> vx
AXIS_LATERAL = 0          # left stick horizontal  -> vy
AXIS_YAW = 2              # right stick horizontal -> wz
GAMEPAD_DEADZONE = 0.1    # ignore small stick drift
BUTTON_RESET = 9          


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


def draw_heightmap(scn, points):
    scn.ngeom = len(points)
    for i, p in enumerate(points):
        mujoco.mjv_initGeom(
            scn.geoms[i],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.02, 0.02, 0.02]),
            np.asarray(p, dtype=np.float64),
            np.eye(3).ravel(),
            np.array([1.0, 0.0, 0.0, 0.7], dtype=np.float32),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True,
                        help='Path to a saved policy, e.g. policies/20260716-142821')
    parser.add_argument('--scene', type=str, default=EnvironmentConfig.filename)
    parser.add_argument('--terrain-seed', type=int, default=0,
                        help='Seed for box-terrain tile randomization (box scenes only)')
    args = parser.parse_args()

    env = UnitreeGo2Env(environment_config=EnvironmentConfig(filename=args.scene))
    n_tiles = apply_random_terrain(env, seed=args.terrain_seed)
    if n_tiles:
        print(f'randomized {n_tiles} terrain tiles (seed {args.terrain_seed})')
    policy_fn = load_policy(args.policy, env)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)

    # Shared command holder, read each frame and pinned into the env.
    command = {'value': jnp.zeros(3), 'reset': False}

    # --- OLD KEYBOARD CONTROL (kept for reference; replaced by gamepad below) ---
    # def key_callback(keycode):
    #     key = chr(keycode) if 0 < keycode < 0x110000 else ''
    #     delta = jnp.zeros(3)
    #     if key == 'W':
    #         delta = jnp.array([CMD_STEP[0], 0.0, 0.0])
    #     elif key == 'S':
    #         delta = jnp.array([-CMD_STEP[0], 0.0, 0.0])
    #     elif key == 'Q':
    #         delta = jnp.array([0.0, CMD_STEP[1], 0.0])
    #     elif key == 'E':
    #         delta = jnp.array([0.0, -CMD_STEP[1], 0.0])
    #     elif key == 'A':
    #         delta = jnp.array([0.0, 0.0, CMD_STEP[2]])
    #     elif key == 'D':
    #         delta = jnp.array([0.0, 0.0, -CMD_STEP[2]])
    #     elif keycode == 32:  # space
    #         command['value'] = jnp.zeros(3)
    #         return
    #     elif key == 'R':
    #         command['reset'] = True
    #         return
    #     command['value'] = jnp.clip(command['value'] + delta, -CMD_MAX, CMD_MAX)

    pygame.init()
    pygame.joystick.init()
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f'gamepad connected: {joystick.get_name()}')
    else:
        print('no controller connected')

    def read_gamepad():
        if joystick is None:
            return jnp.zeros(3)
        pygame.event.pump()  # refresh axis/button state each frame
        fwd = -joystick.get_axis(AXIS_FORWARD)   # stick up   -> +vx
        lat = -joystick.get_axis(AXIS_LATERAL)   # stick left -> +vy
        yaw = -joystick.get_axis(AXIS_YAW)       # stick left -> +wz
        raw = np.array([fwd, lat, yaw], dtype=np.float32)
        raw = np.where(np.abs(raw) < GAMEPAD_DEADZONE, 0.0, raw)  # deadzone
        raw = np.clip(raw, -1.0, 1.0)
        if joystick.get_button(BUTTON_RESET):
            command['reset'] = True
        return jnp.asarray(raw) * CMD_MAX  # scale [-1,1] -> command range

    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)
    dt = env.dt
    jit_heightmap_points = jax.jit(env._heightmap_points)

    with mujoco.viewer.launch_passive(
        mj_model, mj_data, 
    ) as viewer:
        act_rng = jax.random.PRNGKey(1)
        last_cmd = None
        while viewer.is_running():
            step_start = time.time()

            command['value'] = read_gamepad()

            if command['reset']:
                rng, sub = jax.random.split(rng)
                state = jit_reset(sub)
                command['reset'] = False

            state.info['command'] = command['value']
            state.info['steps_until_next_command'] = jnp.int32(1_000_000)

            act_rng, sub = jax.random.split(act_rng)
            action, _ = policy_fn(state.obs, sub)
            state = jit_step(state, action)

            # Mirror MJX state into the CPU model for rendering.
            mj_data.qpos[:] = np.asarray(state.data.qpos)
            mj_data.qvel[:] = np.asarray(state.data.qvel)
            mujoco.mj_forward(mj_model, mj_data)
            draw_heightmap(viewer.user_scn, np.asarray(jit_heightmap_points(state.data)))
            viewer.sync()

            cmd = tuple(round(float(c), 2) for c in command['value'])
            if cmd != last_cmd:
                print(f'command [vx, vy, wz] = {cmd}')
                last_cmd = cmd

            wait = dt - (time.time() - step_start)
            if wait > 0:
                time.sleep(wait)


if __name__ == '__main__':
    main()
