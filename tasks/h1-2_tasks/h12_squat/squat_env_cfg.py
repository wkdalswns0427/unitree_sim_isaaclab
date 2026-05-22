"""H1-2 squat task: track a randomly resampled pelvis-height target without falling.

Action subset matches h12_velocity (legonly): 13 DOF (12 legs + torso).  Arms
are held by their actuator config.  The "command" is a target pelvis height
sampled uniformly in ``height_range`` and resampled every few seconds.

Design rationale
----------------
* Squatting is a height-tracking task, not a velocity-tracking task.  We
  inherit ``LocomotionVelocityRoughEnvCfg`` only for its sim/scene plumbing
  and replace ``CommandsCfg``, ``ObservationsCfg``, and ``RewardsCfg``
  wholesale with squat-specific versions.
* Lower-bound ``root_height`` termination is dropped to 0.30 m so deep
  squats (target ~0.50 m pelvis height + bob) do not trip it.
* ``track_pelvis_height_exp`` Gaussian reward has a narrow std (0.10 m) so
  the policy gets a strong shaping signal even when it's "kind of close" to
  the target.  Pair with a small L1 penalty to close the last few cm.
* ``both_feet_grounded_bonus`` discourages the one-foot-balance failure
  mode early policies fall into.  Pair with ``feet_slide`` so the bonus
  cannot be farmed by skating.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
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

from .mdp import commands as squat_commands
from .mdp import curriculums as squat_curriculums
from .mdp import observations as squat_obs
from .mdp import rewards as squat_rewards


SQUAT_JOINT_NAMES = [
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


# ── Commands ────────────────────────────────────────────────────────────────

@configclass
class H12SquatCommandsCfg:
    """Alternating pelvis-height target: 3-second hold at each end.

    Each env oscillates between ``stand_high = 1.00`` m and ``squat_low =
    0.55`` m, holding each pose for exactly 3 seconds before the target
    flips.  Initial phase is randomized per-env so PPO sees both transition
    directions in every minibatch.
    """

    target_height = squat_commands.UniformPelvisHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 3.0),
        # Initial range matches the squat-depth curriculum's starting point
        # (start_low=0.92).  The curriculum widens this toward (0.55, 1.00)
        # as training episodes accumulate.
        height_range=(0.92, 1.00),
        mode="alternating",
        debug_vis=False,
    )


# ── Observations ─────────────────────────────────────────────────────────────

@configclass
class H12SquatObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=base_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        target_height = ObsTerm(
            func=squat_obs.target_pelvis_height,
            params={"command_name": "target_height"},
        )
        pelvis_height = ObsTerm(
            func=squat_obs.pelvis_height,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SQUAT_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SQUAT_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ── Rewards ──────────────────────────────────────────────────────────────────

@configclass
class H12SquatRewardsCfg:
    # Task: track target pelvis height
    track_pelvis_height_exp = RewTerm(
        func=squat_rewards.track_pelvis_height_exp,
        weight=3.0,
        params={"command_name": "target_height", "std": 0.10},
    )
    pelvis_height_l1 = RewTerm(
        func=squat_rewards.pelvis_height_l1,
        weight=-0.5,
        params={"command_name": "target_height"},
    )

    # Safety
    termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=-200.0)

    # Stay in place — no horizontal drift, no spinning
    lin_vel_xy_l2 = RewTerm(func=squat_rewards.lin_vel_xy_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-0.10)
    ang_vel_z_l2 = RewTerm(func=squat_rewards.ang_vel_z_l2, weight=-0.10)

    # Stay upright
    flat_orientation_l2 = RewTerm(func=base_mdp.flat_orientation_l2, weight=-2.5)

    # Encourage both feet loaded during squat — closes the one-foot-balance
    # local optimum.  Use feet_slide to make sure the policy isn't farming
    # the bonus by skating in place.
    both_feet_grounded = RewTerm(
        func=squat_rewards.both_feet_grounded_bonus,
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

    # Joint regularization
    dof_pos_limits = RewTerm(
        func=base_mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_POSTURE_JOINTS)},
    )
    # Heavy anchor on torso_joint so the policy cannot twist the trunk during
    # the squat — we want the robot to face forward the whole time.
    joint_deviation_torso = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )

    # Smoothness
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-0.005)
    dof_acc_l2 = RewTerm(
        func=base_mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINTS)},
    )


# ── Terminations ─────────────────────────────────────────────────────────────

@configclass
class H12SquatTerminationsCfg:
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
    # Allow deep squats — min pelvis height = 0.30 m (~30 cm below stand pose).
    root_height = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.30}
    )


# ── Curriculum ───────────────────────────────────────────────────────────────

@configclass
class H12SquatCurriculumCfg:
    """Replaces the inherited terrain-only curriculum cfg.

    ``squat_depth`` linearly widens the squat depth from a shallow
    ``(0.92, 1.00)`` dip toward the full ``(0.55, 1.00)`` range over the
    first ~50k episode resets (~few hundred PPO iterations at typical env
    counts).  Tune ``progression_resets`` to your training budget.
    """

    squat_depth = CurrTerm(
        func=squat_curriculums.progressive_squat_depth,
        params={
            "start_low": 0.92,
            "end_low": 0.55,
            "stand_high": 1.00,
            "progression_resets": 50000,
        },
    )


# ── Env cfg ──────────────────────────────────────────────────────────────────

@configclass
class H12SquatEnvCfg(LocomotionVelocityRoughEnvCfg):
    commands: H12SquatCommandsCfg = H12SquatCommandsCfg()
    observations: H12SquatObservationsCfg = H12SquatObservationsCfg()
    rewards: H12SquatRewardsCfg = H12SquatRewardsCfg()
    terminations: H12SquatTerminationsCfg = H12SquatTerminationsCfg()
    curriculum: H12SquatCurriculumCfg = H12SquatCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # ── H1-2 physical configuration ──────────────────────────────────────
        robot_cfg = H12RobotPresets.h12_27dof_inspire_wholebody_floating(
            init_pos=(0.0, 0.0, 1.0),
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
            self.scene.height_scanner = None  # not used for squat
        # Flat ground — squat doesn't need terrain curriculum.  We replaced
        # the inherited `LocomotionVelocityRoughEnvCfg.curriculum` cfg with
        # `H12SquatCurriculumCfg` (no terrain_levels term) so nothing to wipe.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # ── Action subset (13 DOF — legs + torso) ────────────────────────────
        self.actions.joint_pos.joint_names = SQUAT_JOINT_NAMES
        self.actions.joint_pos.scale = 0.5
        self.actions.joint_pos.preserve_order = True

        # ── Randomization (H1-2 reference) ───────────────────────────────────
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        # No spawn-yaw randomization — the robot must face forward (+x) the
        # entire episode.  Tiny x/y jitter is fine; full-yaw randomization
        # confuses the height-tracking signal because the policy then has to
        # also learn a yaw-invariant standing posture.
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }


@configclass
class H12SquatEnvCfg_PLAY(H12SquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 60.0
        self.actions.joint_pos.scale = 1.0
        # Same 3-second hold schedule as training for visual verification.
        self.commands.target_height.resampling_time_range = (3.0, 3.0)
        self.commands.target_height.height_range = (0.55, 1.00)
        self.commands.target_height.mode = "alternating"
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
