from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_rough_env_cfg,
  
  unitree_go2_rough_blind_env_cfg,

  unitree_go2_rough_medium_env_cfg,
  unitree_go2_rough_medium_2_env_cfg,

  unitree_go2_rough_hard_env_cfg,

  unitree_go2_all_terrain_env_cfg

)
from .rl_cfg import unitree_go2_ppo_runner_cfg


def main() -> None:
  print("Hello from unitree-vision-rl!")

### tasks that build entire scene and environment for training

register_mjlab_task(
  task_id="Velocity-Rough-Unitree-Go2",
  env_cfg=unitree_go2_rough_env_cfg(),
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Velocity-Flat-Unitree-Go2",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Velocity-Rough-Blind-Unitree-Go2",
  env_cfg=unitree_go2_rough_blind_env_cfg(),
  play_env_cfg=unitree_go2_rough_blind_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Velocity-Rough-Medium-Unitree-Go2",
  env_cfg=unitree_go2_rough_medium_env_cfg(),
  play_env_cfg=unitree_go2_rough_medium_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Velocity-Rough-Hard-Unitree-Go2",
  env_cfg=unitree_go2_rough_hard_env_cfg(),
  play_env_cfg=unitree_go2_rough_hard_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Velocity-Rough-Medium-2-Unitree-Go2",
  env_cfg=unitree_go2_rough_medium_2_env_cfg(),
  play_env_cfg=unitree_go2_rough_medium_2_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="All-Terrains-Test-Unitree-Go2",
  env_cfg=unitree_go2_all_terrain_env_cfg(),
  play_env_cfg=unitree_go2_all_terrain_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)