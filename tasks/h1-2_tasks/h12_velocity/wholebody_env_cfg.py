"""H1-2 velocity task — legacy 21-DOF wholebody action subset.

Action / observation subset is legs (12) + torso (1) + arms (8) = 21, matching
the configuration used for the 2026-05-08 and 2026-05-15 training runs.
Rewards, terminations, commands, and randomization inherit from
``rough_env_cfg.H12Rewards`` / ``H12RoughEnvCfg``.

Use this config when you want the policy to actively swing arms for momentum
compensation.  The current lower-body-only config (legs + torso = 13 DOF) is in
``rough_env_cfg.py`` and is the recommended starting point — arm wobble there
is suppressed by the actuator config holding arms at default.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .flat_env_cfg import H12FlatEnvCfg
from .rough_env_cfg import H12RoughEnvCfg


WHOLEBODY_JOINT_NAMES = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]


def _apply_wholebody_subset(cfg) -> None:
    cfg.actions.joint_pos.joint_names = WHOLEBODY_JOINT_NAMES
    cfg.observations.policy.joint_pos.params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=WHOLEBODY_JOINT_NAMES, preserve_order=True)
    }
    cfg.observations.policy.joint_vel.params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=WHOLEBODY_JOINT_NAMES, preserve_order=True)
    }


@configclass
class H12WholebodyRoughEnvCfg(H12RoughEnvCfg):
    """21-DOF wholebody action subset, rough terrain."""

    def __post_init__(self):
        super().__post_init__()
        _apply_wholebody_subset(self)


@configclass
class H12WholebodyFlatEnvCfg(H12FlatEnvCfg):
    """21-DOF wholebody action subset, flat terrain."""

    def __post_init__(self):
        super().__post_init__()
        _apply_wholebody_subset(self)


@configclass
class H12WholebodyFlatEnvCfg_PLAY(H12WholebodyFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 60.0
        self.actions.joint_pos.scale = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
