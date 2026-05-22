"""H1-2 balanced-standing task: hold default pose at a fixed pelvis height
without falling.

Compared to ``Isaac-H12-Squat-v0`` this is the *trivial* base case — the
target pelvis height is a constant (~1.05 m) instead of a resampled
command, there is no curriculum, and the reward sheet is dominated by
"zero motion + upright + at-height" terms.  It exists to give the policy
a clean baseline for passive balance before layering in any commanded
motion.

Action subset is the same 13-DOF legs+torso slice used by the velocity-
legonly and squat tasks; arms are held by their actuator config.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as base_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from tasks.common_config import H12RobotPresets

from .mdp import rewards as stand_rewards


# Same lower-body + torso slice used by h12_velocity (legonly) and h12_squat.
STAND_JOINT_NAMES = [
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
]

LOWER_BODY_JOINTS = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]
ARM_POSTURE_JOINTS = [".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]

# Target standing pelvis height (m).  H1-2 with the default joint pose
# settles around 1.05 m; we anchor the reward there.
STAND_PELVIS_HEIGHT = 1.05


# ── Commands ────────────────────────────────────────────────────────────────
#
# Standing has no commanded motion, so we drop the velocity command term
# entirely.  The inherited ``LocomotionVelocityRoughEnvCfg`` defines a
# ``base_velocity`` command; we override the whole commands cfg with an
# empty one.

@configclass
class H12StandCommandsCfg:
    """No commands — standing is a pure regulation task."""
    pass


# ── Observations ─────────────────────────────────────────────────────────────

@configclass
class H12StandObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=base_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=STAND_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=STAND_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ── Rewards ──────────────────────────────────────────────────────────────────

@configclass
class H12StandRewardsCfg:
    # Stay alive — small per-step positive so termination genuinely costs.
    alive = RewTerm(func=base_mdp.is_alive, weight=1.0)
    termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=-200.0)

    # Hold pelvis at the standing height (L2 on `z - target`).
    base_height_l2 = RewTerm(
        func=base_mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": STAND_PELVIS_HEIGHT},
    )

    # Stay upright.
    flat_orientation_l2 = RewTerm(func=base_mdp.flat_orientation_l2, weight=-5.0)

    # No motion: penalize every velocity axis.
    lin_vel_xy_l2 = RewTerm(func=stand_rewards.lin_vel_xy_l2, weight=-2.0)
    lin_vel_z_l2 = RewTerm(func=base_mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-0.5)
    ang_vel_z_l2 = RewTerm(func=stand_rewards.ang_vel_z_l2, weight=-1.0)

    # Both feet loaded — closes the "single-foot teeter" local optimum.
    both_feet_grounded = RewTerm(
        func=stand_rewards.both_feet_grounded_bonus,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "force_threshold": 20.0,
        },
    )
    feet_slide = RewTerm(
        func=base_mdp.feet_slide,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Joint regularization — anchor hips, torso, and arms to defaults.
    dof_pos_limits = RewTerm(
        func=base_mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_POSTURE_JOINTS)},
    )
    joint_deviation_torso = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )

    # Smoothness.
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    dof_acc_l2 = RewTerm(
        func=base_mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINTS)},
    )


# ── Terminations ─────────────────────────────────────────────────────────────

@configclass
class H12StandTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(
        func=base_mdp.bad_orientation, params={"limit_angle": 0.9}
    )
    root_height = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.55}
    )


# ── Curriculum ───────────────────────────────────────────────────────────────

@configclass
class H12StandCurriculumCfg:
    """Empty — passive standing needs no curriculum."""
    pass


# ── Env cfg ──────────────────────────────────────────────────────────────────

@configclass
class H12StandEnvCfg(LocomotionVelocityRoughEnvCfg):
    commands: H12StandCommandsCfg = H12StandCommandsCfg()
    observations: H12StandObservationsCfg = H12StandObservationsCfg()
    rewards: H12StandRewardsCfg = H12StandRewardsCfg()
    terminations: H12StandTerminationsCfg = H12StandTerminationsCfg()
    curriculum: H12StandCurriculumCfg = H12StandCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # ── H1-2 physical configuration ──────────────────────────────────────
        robot_cfg = H12RobotPresets.h12_27dof_inspire_wholebody_floating(
            init_pos=(0.0, 0.0, 1.05),
            init_rot=(1.0, 0.0, 0.0, 0.0),
        )
        robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 4
        robot_cfg.actuators["feet"].effort_limit_sim = {
            ".*_ankle_pitch_joint": 45.0,
            ".*_ankle_roll_joint": 45.0,
        }
        robot_cfg.actuators["feet"].stiffness = 20.0
        robot_cfg.actuators["feet"].damping = 2.5
        self.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner is not None:
            self.scene.height_scanner = None

        # Flat ground — standing on rough terrain is a separate problem.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # ── Action subset (13 DOF — legs + torso) ────────────────────────────
        self.actions.joint_pos.joint_names = STAND_JOINT_NAMES
        self.actions.joint_pos.scale = 0.5
        self.actions.joint_pos.preserve_order = True

        # Shorter episodes — standing failure is fast, no need for 20-s rollouts.
        self.episode_length_s = 10.0

        # ── Randomization / external disturbances ────────────────────────────
        # Standing has no momentum reserve, so pushes are scaled smaller and
        # made more frequent than the walking defaults (±0.5 m/s every
        # 10-15 s in the base velocity cfg).
        self.events.push_robot.interval_range_s = (4.0, 8.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
        }

        # Sustained force/torque on the torso during each episode — like
        # a steady shove or carrying an off-center load.  Mode is "reset",
        # so the same force persists for the whole episode and is rerolled
        # at the next reset.
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.base_external_force_torque.params["force_range"] = (-15.0, 15.0)
        self.events.base_external_force_torque.params["torque_range"] = (-3.0, 3.0)

        # Mass + CoM domain randomization (startup, per-env, fixed for the
        # whole episode life).  Keep ranges modest — H1-2 base is ~45 kg.
        self.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-3.0, 3.0)
        self.events.base_com.params["asset_cfg"].body_names = ["torso_link"]
        self.events.base_com.params["com_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.01, 0.01),
        }

        # Joint reset: small jitter around defaults instead of the walking
        # cfg's wide (0.5, 1.5) range — standing starts from a known pose.
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.0),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        }


@configclass
class H12StandEnvCfg_PLAY(H12StandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 60.0
        self.actions.joint_pos.scale = 1.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
