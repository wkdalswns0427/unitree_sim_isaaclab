"""H1-2 velocity task — symmetric mirrored bipedal walking.

Failure mode this version targets: prior policy converged to a "pendulum
leg" — one leg held bent and swinging in air, the other locked straight on
the ground, scooting forward without ever taking a real step.

Three structural fixes vs. the prior reward set:

1. Replaced ``feet_air_time_positive_biped`` with event-based
   ``feet_air_time``.  The biped variant pays ``min(left_in_mode_time,
   right_in_mode_time)`` during single-stance, where ``in_mode_time`` is
   ``contact_time`` for the loaded foot and ``air_time`` for the lifted
   foot.  A permanently-suspended leg + permanently-loaded leg saturates
   both at the threshold without any touchdown — the loophole the policy
   exploited.  Event-based ``feet_air_time`` only pays
   ``(last_air_time - threshold)`` on the air→contact transition, so a
   leg that never lands earns zero.

2. ``sustained_single_foot_contact`` penalizes parking on one loaded foot
   longer than ``max_single_stance_time`` — the other half of the
   anti-pendulum constraint.

3. ``leg_antisymmetry_penalty`` enforces ``(left_delta + right_delta)²``
   ≈ 0 in the sagittal plane (hip-pitch / knee / ankle-pitch), driving
   the legs into mirror-image anti-phase motion.  Without an explicit
   mirror constraint the policy is free to pick any asymmetric posture
   as a local optimum.

Per-side ``swing_knee_flexion`` is preserved — only the swinging leg
collects, so a one-leg-pendulum policy captures half the signal vs. an
alternating gait, breaking the local-optimum tie.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from tasks.common_config import H12RobotPresets
from .mdp import rewards as h12_rewards


LOCOMOTION_JOINT_NAMES = [
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

LOWER_BODY_JOINTS = [
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
]

ARM_POSTURE_JOINTS = [
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
]

# Sagittal-plane joints for mirror-symmetry enforcement.  Order matters:
# leg_antisymmetry_penalty unpacks as
# (left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle).
MIRROR_LEG_JOINTS = [
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
]


@configclass
class H12Rewards(RewardsCfg):
    # ── Safety ───────────────────────────────────────────────────────────────
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # Disable lin_vel_z penalty entirely (H1 baseline).  Walking has a natural
    # ~5 cm CoM bob; penalizing it forces a stiff drag gait.
    lin_vel_z_l2 = None

    # ── Tracking ─────────────────────────────────────────────────────────────
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # ── Gait alternation ─────────────────────────────────────────────────────
    # Event-based: pays (last_air_time - threshold) only on the air→contact
    # transition.  A leg held permanently in air earns zero — closes the
    # "pendulum leg" loophole that feet_air_time_positive_biped allows.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    # Punish parking on one loaded foot longer than max_single_stance_time —
    # the other half of the anti-pendulum constraint.
    no_single_foot_park = RewTerm(
        func=h12_rewards.sustained_single_foot_contact,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "max_single_stance_time": 0.5,
            "force_threshold": 20.0,
        },
    )
    # Mirror-symmetry: penalizes (left + right) deviation in sagittal plane,
    # forcing hips/knees/ankles into anti-phase motion.  Without this term
    # the policy can pick an asymmetric posture as a local optimum.
    leg_mirror = RewTerm(
        func=h12_rewards.leg_antisymmetry_penalty,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=MIRROR_LEG_JOINTS, preserve_order=True),
        },
    )
    # Anti-hop: punish frames where neither foot is loaded while the robot
    # is commanded to move.  At higher cadences the policy can shortcut to
    # a hop ("jump to next step") which the per-foot air-time term doesn't
    # forbid.  Pairs with no_single_foot_park: together they require
    # exactly-one-foot-loaded during locomotion.
    flight_phase = RewTerm(
        func=h12_rewards.flight_phase_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "force_threshold": 20.0,
        },
    )
    # Anti-drag: punish foot motion while in contact.
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # Knee-driven swing: directly pays the policy for bending the swing-phase
    # knee.  Target 0.5 rad ≈ 29° gives clear toe-clearance without forcing a
    # high-stepping march.  Without this term the policy converges to hip-hike
    # + straight-leg drag.
    swing_knee_flexion_left = RewTerm(
        func=h12_rewards.swing_knee_flexion,
        weight=0.3,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names="left_knee_joint"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="left_ankle_roll_link"),
            "min_flexion": 0.2,
            "target_flexion": 0.5,
            "force_threshold": 20.0,
        },
    )
    swing_knee_flexion_right = RewTerm(
        func=h12_rewards.swing_knee_flexion,
        weight=0.3,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names="right_knee_joint"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="right_ankle_roll_link"),
            "min_flexion": 0.2,
            "target_flexion": 0.5,
            "force_threshold": 20.0,
        },
    )

    # ── Joint limits ─────────────────────────────────────────────────────────
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )

    # ── Posture regularization ───────────────────────────────────────────────
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    # Strong arm anchor — no active arm reward, only this L1 deviation pull
    # toward the default rest pose.  Keeps arms quiet without freezing them.
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_POSTURE_JOINTS)},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )
    # Disabled: stand-still bonus competes with the velocity tracking signal
    # at low commanded speeds.
    stand_still = None


@configclass
class H12RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: H12Rewards = H12Rewards()

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
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # 21-DOF action subset (12 leg + torso + 8 arm).
        self.actions.joint_pos.joint_names = LOCOMOTION_JOINT_NAMES
        self.actions.joint_pos.scale = 0.5
        self.actions.joint_pos.preserve_order = True
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=LOCOMOTION_JOINT_NAMES, preserve_order=True)
        }
        self.observations.policy.joint_vel.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=LOCOMOTION_JOINT_NAMES, preserve_order=True)
        }

        # ── Randomization (H1 reference) ─────────────────────────────────────
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # ── Light stability penalties ────────────────────────────────────────
        # Kept gentle so they do not dominate the gait incentives.
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINTS)
        self.rewards.dof_torques_l2.weight = 0.0  # H1 baseline: torque penalty off

        # ── Commands ─────────────────────────────────────────────────────────
        # Allow the standing range so the policy can learn from rest; H1
        # baseline does not force a non-zero forward command.
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # ── Terminations ─────────────────────────────────────────────────────
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        # Cut episodes early when the robot tilts past ~50° or sinks too low,
        # so the policy can't farm lie-down survival reward.
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation, params={"limit_angle": 0.9}
        )
        self.terminations.root_height = DoneTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": 0.6}
        )

        # Increase upright posture penalty so it pushes back against tilt.
        self.rewards.flat_orientation_l2.weight = -2.5


@configclass
class H12RoughEnvCfg_PLAY(H12RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
