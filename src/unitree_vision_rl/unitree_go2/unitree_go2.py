from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_HERE = Path(__file__).parent
GO2_XML: Path = _HERE / "xml" / "unitree_go2.xml"
assert GO2_XML.exists()


# returning mj spec from xml file. robot model
def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2_XML))
  return spec

# actuator config.
STIFFNESS = 35.0  # kp
DAMPING = 0.5  # kv
HIP_EFFORT_LIMIT = 23.7  # abduction + hip forcerange
KNEE_EFFORT_LIMIT = 45.43  # knee forcerange

GO2_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=HIP_EFFORT_LIMIT,
)
GO2_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=KNEE_EFFORT_LIMIT,
)

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.3),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    ".*_hip_joint": 0.0,
    ".*_thigh_joint": 0.9,
    ".*_calf_joint": -1.8,
  },
  joint_vel={".*": 0.0},
)

# collision config
_FOOT_REGEX = r"^.*_foot_collision$"
_COLLISION_REGEX = r"^.*_collision$"

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_FOOT_REGEX,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.8, 0.02, 0.01),
  solref=(0.01, 1.0),
  solimp=(0.9, 0.95, 0.022),
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_COLLISION_REGEX,),
  contype=1,
  conaffinity=1,
  condim={_FOOT_REGEX: 3, _COLLISION_REGEX: 1},
  priority={_FOOT_REGEX: 1, ".*": 0},
  friction={_FOOT_REGEX: (0.8, 0.02, 0.01)},
  solref=(0.01, 1.0),
  solimp={_FOOT_REGEX: (0.9, 0.95, 0.022)},
)


# final config
GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(GO2_HIP_ACTUATOR_CFG, GO2_KNEE_ACTUATOR_CFG),
  soft_joint_pos_limit_factor=0.9,
)


# func bringing it all together
def get_go2_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_ARTICULATION,
  )


GO2_ACTION_SCALE: dict[str, float] = {}
for _a in GO2_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  assert _a.effort_limit is not None
  for _n in _a.target_names_expr:
    GO2_ACTION_SCALE[_n] = 0.25 * _a.effort_limit / _a.stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity
  robot = Entity(get_go2_robot_cfg())
  viewer.launch(robot.spec.compile())
