"""
Unitree Go2 Environment Configuration:
"""

import jax
import jax.numpy as jnp
import flax.struct


@flax.struct.dataclass
class RewardConfig:
    # Rewards:
    tracking_linear_velocity: float = 1.5
    tracking_angular_velocity: float = 0.75
    # Orientation Regularization Terms:
    orientation_regularization: float = -0.95
    linear_z_velocity: float = -0.5
    angular_xy_velocity: float = -0.05
    # Energy Regularization Terms:
    torque: float = -2e-4
    action_rate: float = -0.01
    acceleration: float = -2.5e-7
    # Auxilary Terms:
    stand_still: float = -1.0
    termination: float = -1.0
    unwanted_contact: float = -0.5
    # Gait Reward Terms:
    foot_slip: float = -0.1
    air_time: float = 0.45
    foot_clearance: float = 1.0
    gait_variance: float = -0.5
    # Gait Hyperparameters:
    target_air_time: float = 0.65
    mode_time: float = 0.3
    command_threshold: float = 0.0
    velocity_threshold: float = 0.5
    # Foot Clearance Reward Terms:
    target_foot_height: float = 0.13
    foot_clearance_velocity_scale: float = 2.0
    foot_clearance_sigma: float = 0.05
    # Hyperparameter for exponential kernel:
    kernel_sigma: float = 0.25


# noise for senses
@flax.struct.dataclass
class NoiseConfig:
    joint_position: float = 0.05
    joint_velocity: float = 1.5
    gyroscope: float = 0.2
    gravity_vector: float = 0.05
    heightmap: float = 0.0  # 0 for now, but will add noise for future experiments


# balancing robot through disturbances (shoves)
@flax.struct.dataclass
class DisturbanceConfig:
    wait_times: list[float] = flax.struct.field(
        default_factory=lambda: [1.0, 3.0],  # how long between each one
    )
    durations: list[float] = flax.struct.field(
        default_factory=lambda: [0.05, 0.2],  # how long it lasts
    )
    magnitudes: list[float] = flax.struct.field(
        default_factory=lambda: [0.0, 3.0],  # how much
    )


@flax.struct.dataclass
class CommandConfig:
    command_range: jax.Array = flax.struct.field(  # max [vx, vy, wz]
        default_factory=lambda: jnp.array([1.5, 1.0, 1.2]),
    )
    single_command_probability: float = 0.0  # chance to isolate one axis
    command_mask_probability: float = (
        0.9  # chance its a real command that it could encounter
    )
    command_frequency: list[float] = flax.struct.field(  # secs between commands
        default_factory=lambda: [1.0, 5.0],
    )


@flax.struct.dataclass
class EnvironmentConfig:
    filename: str = "scene_mjx_vendor_torque_steps.xml"
    impl: str = "warp"
    action_scale: float = 0.5
    control_timestep: float = 0.02
    optimizer_timestep: float = 0.004
    naconmax: int = 8 * 8192
    # Max constraint rows per world. Flat ground fit in 60, but stepped/rough
    # hfield produces more simultaneous foot contacts and overflowed (~73),
    # dropping contacts. Headroom above peak; dynamic parkour contact needs it.
    njmax: int = 128
    # Half-width (m) of the box the robot spawns in, centered on the terrain.
    # Widen this to sample more of the rough field per episode. The hfield is
    # +/-10 m, so 4.0 leaves a 6 m margin before the edge.
    spawn_radius: float = 4.0

    # heightmap env config
    heightmap_enabled: bool = True
    heightmap_rows: int = 11
    heightmap_cols: int = 7
    heightmap_spacing: float = 0.1
    heightmap_clip: float = 0.5
    ray_origin_margin: float = 1.0
    spawn_clearance: float = 0.0
